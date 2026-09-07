#include "mlek/use_case/rhythmformer/RhythmFormerProcessing.hpp"

#include "mlek/log/log_macros.h"

#if (defined(__ARM_FEATURE_DSP) && (__ARM_FEATURE_DSP == 1))
#include "arm_math.h"
#endif

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace arm {
namespace app {

    RhythmFormerPreProcess::FrameRing RhythmFormerPreProcess::m_frameRing RHYTHMFORMER_DRAM_ATTR;
    RhythmFormerPostProcess::BvpHistory RhythmFormerPostProcess::m_bvpHistory RHYTHMFORMER_DRAM_ATTR;
    size_t RhythmFormerPostProcess::m_bvpHistoryCount;

    float    RhythmFormerPostProcess::m_avgBatch[kAvgWindowCycles];
    float    RhythmFormerPostProcess::m_avgConfidenceBatch[kAvgWindowCycles];
    size_t   RhythmFormerPostProcess::m_avgBatchSize = 0;
    size_t   RhythmFormerPostProcess::m_avgBatchWriteIdx = 0;
    float    RhythmFormerPostProcess::m_smoothedBpm = 0.0f;
    float    RhythmFormerPostProcess::m_avgConfidence = 0.0f;
    uint32_t RhythmFormerPostProcess::m_msSinceLastDisplayUpdate = 0;
    float    RhythmFormerPostProcess::m_publishedHr = 0.0f;

    namespace {

        /* Safety floor for ComputeMeanAndInvStd's std-dev divide;
         * BuildChunkTensor no longer standardizes per-chunk. */
        constexpr float kDiffEps = 1e-7f;

        /** Fast 1D coordinate and weight mapping for bilinear interpolation */
        struct GridCoord {
            uint16_t idx0;
            uint16_t idx1;
            float fx;
            float invFx;
        };

        /** Optimized bilinear resize from center-cropped region to dstSize x dstSize */
        void BilinearResizeSquare(const uint8_t* src,
                                  size_t srcW,
                                  size_t srcH,
                                  size_t cropSide,
                                  size_t cropX,
                                  size_t cropY,
                                  float* dst,
                                  size_t dstSize)
        {
            const float scale = static_cast<float>(cropSide) / static_cast<float>(dstSize);

            // Precompute X coordinate mappings for the row
            GridCoord xMap[kRhythmFormerImgSize];
            for (size_t dx = 0; dx < dstSize; ++dx) {
                const float srcX = static_cast<float>(cropX) + (static_cast<float>(dx) + 0.5f) * scale - 0.5f;
                const float clampedX = std::max(0.0f, std::min(static_cast<float>(srcW - 1), srcX));
                const size_t x0 = static_cast<size_t>(clampedX);
                const size_t x1 = std::min(x0 + 1, srcW - 1);
                const float fx = clampedX - static_cast<float>(x0);

                xMap[dx].idx0  = static_cast<uint16_t>(x0);
                xMap[dx].idx1  = static_cast<uint16_t>(x1);
                xMap[dx].fx    = fx;
                xMap[dx].invFx = 1.0f - fx;
            }

            for (size_t dy = 0; dy < dstSize; ++dy) {
                const float srcY = static_cast<float>(cropY) + (static_cast<float>(dy) + 0.5f) * scale - 0.5f;
                const float clampedY = std::max(0.0f, std::min(static_cast<float>(srcH - 1), srcY));
                const size_t y0 = static_cast<size_t>(clampedY);
                const size_t y1 = std::min(y0 + 1, srcH - 1);
                const float fy = clampedY - static_cast<float>(y0);
                const float invFy = 1.0f - fy;

                const uint8_t* row0 = src + (y0 * srcW) * kRhythmFormerRgbChannels;
                const uint8_t* row1 = src + (y1 * srcW) * kRhythmFormerRgbChannels;
                float* dstRow = dst + (dy * dstSize) * kRhythmFormerRgbChannels;

                for (size_t dx = 0; dx < dstSize; ++dx) {
                    const size_t x0_ch = xMap[dx].idx0 * kRhythmFormerRgbChannels;
                    const size_t x1_ch = xMap[dx].idx1 * kRhythmFormerRgbChannels;

                    const float w00 = xMap[dx].invFx * invFy;
                    const float w01 = xMap[dx].fx    * invFy;
                    const float w10 = xMap[dx].invFx * fy;
                    const float w11 = xMap[dx].fx    * fy;

                    const size_t dst_idx = dx * kRhythmFormerRgbChannels;

                    for (size_t c = 0; c < kRhythmFormerRgbChannels; ++c) {
                        const float p00 = static_cast<float>(row0[x0_ch + c]);
                        const float p01 = static_cast<float>(row0[x1_ch + c]);
                        const float p10 = static_cast<float>(row1[x0_ch + c]);
                        const float p11 = static_cast<float>(row1[x1_ch + c]);

                        dstRow[dst_idx + c] = p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11;
                    }
                }
            }
        }

        /** Optimized mean and standard deviation computation using CMSIS-DSP if available */
        inline void ComputeMeanAndInvStd(const float* data, size_t count, float& outMean, float& outInvStd)
        {
#if (defined(__ARM_FEATURE_DSP) && (__ARM_FEATURE_DSP == 1))
            float mean = 0.f;
            float stdDev = 0.f;
            arm_mean_f32(const_cast<float*>(data), static_cast<uint32_t>(count), &mean);
            arm_std_f32(const_cast<float*>(data), static_cast<uint32_t>(count), &stdDev);
            if (stdDev < kDiffEps) {
                stdDev = kDiffEps;
            }
            outMean = mean;
            outInvStd = 1.0f / stdDev;
#else
            double sum = 0.0;
            double sumSq = 0.0;
            for (size_t i = 0; i < count; ++i) {
                const double d = static_cast<double>(data[i]);
                sum += d;
                sumSq += d * d;
            }
            const double mean = sum / static_cast<double>(count);
            double var = (sumSq / static_cast<double>(count)) - (mean * mean);
            if (var < 0.0) {
                var = 0.0;
            }
            float stdDev = static_cast<float>(std::sqrt(var));
            if (stdDev < kDiffEps) {
                stdDev = kDiffEps;
            }
            outMean = static_cast<float>(mean);
            outInvStd = 1.0f / stdDev;
#endif
        }

        /** Design a 2nd-order Butterworth bandpass via bilinear transform,
         *  pre-warped to [lowHz, highHz], producing 2 SOS sections (2nd-order
         *  lowpass prototype -> 4th-order bandpass).
         *
         *  Built by direct pole placement: the lowpass prototype's single
         *  conjugate pole pair is mapped through the standard LP->BP
         *  substitution to get two bandpass pole pairs, each bilinear-
         *  transformed into its own SOS section. Gain is not separately
         *  normalized -- the peak-detection and power-ratio confidence
         *  metrics used downstream are scale-invariant, so exact unity
         *  passband gain isn't required. */
        void ComputeButter2BpSosCoefs(float fps, float lowHz, float highHz,
                                      RhythmFormerPostProcess::ButterSos& s1,
                                      RhythmFormerPostProcess::ButterSos& s2)
        {
            /* 1. Pre-warp band edges via bilinear transform (s = (z-1)/(z+1),
             *    matching the substitution used in bilinearBiquad below),
             *    compute bandwidth and center frequency as normalized rad/s. */
            const float piFps = static_cast<float>(M_PI) / fps;
            const float wLo = std::tan(lowHz  * piFps);
            const float wHi = std::tan(highHz * piFps);
            const float bw  = wHi - wLo;                           /* bandwidth */
            const float w0  = std::sqrt(wLo * wHi);                 /* center */

            /* 2. One LP prototype pole (its conjugate is handled implicitly:
             *    the two BP poles derived from p pair with the two BP poles
             *    derived from conj(p) to form the two conjugate SOS pairs). */
            const std::complex<float> p(-0.70710678f, 0.70710678f); /* -cos(pi/4), sin(pi/4) */

            /* 3. LP->BP pole transform: roots of s^2 - bw*p*s + w0^2 = 0. */
            const std::complex<float> bwP = bw * p;
            const std::complex<float> disc = bwP * bwP - std::complex<float>(4.0f * w0 * w0, 0.0f);
            const std::complex<float> sqrtDisc = std::sqrt(disc);
            const std::complex<float> sA = (bwP + sqrtDisc) / 2.0f;
            const std::complex<float> sB = (bwP - sqrtDisc) / 2.0f;

            auto bilinearBiquad = [](float cS2, float cS1, float cS0,
                                     float nS2, float nS1, float nS0,
                                     float& b0, float& b1, float& b2,
                                     float& a1, float& a2) {
                /* Bilinear s = (z-1)/(z+1). For denom cS2*s^2 + cS1*s + cS0:
                 *   cS2*(z-1)^2 + cS1*(z-1)(z+1) + cS0*(z+1)^2 = 0
                 * = (cS2 + cS1 + cS0) z^2 + (-2cS2 + 2cS0) z + (cS2 - cS1 + cS0) */
                const float d0 = cS2 + cS1 + cS0;
                const float d1 = -2.0f * cS2 + 2.0f * cS0;
                const float d2 = cS2 - cS1 + cS0;
                a1 = d1 / d0;
                a2 = d2 / d0;
                const float nz2 = nS2 + nS1 + nS0;
                const float nz1 = -2.0f * nS2 + 2.0f * nS0;
                const float nz0 = nS2 - nS1 + nS0;
                b0 = nz2 / d0;
                b1 = nz1 / d0;
                b2 = nz0 / d0;
            };

            /* Section from pole pair {s_pole, conj(s_pole)}:
             *   denom = s^2 - 2*Re(s_pole)*s + |s_pole|^2
             *   numer = bw*s */
            auto sectionFromPole = [&](const std::complex<float>& sPole, RhythmFormerPostProcess::ButterSos& out) {
                const float cS1 = -2.0f * sPole.real();
                const float cS0 = std::norm(sPole); /* |s_pole|^2 */
                bilinearBiquad(1.0f, cS1, cS0,
                               0.0f, bw, 0.0f,
                               out.b0, out.b1, out.b2, out.a1, out.a2);
            };

            sectionFromPole(sA, s1);
            sectionFromPole(sB, s2);
        }

    } /* anonymous namespace */

    /* -------------------------------------------------------------------- */
    /* RhythmFormerPreProcess implementation                                        */
    /* -------------------------------------------------------------------- */

    RhythmFormerPreProcess::RhythmFormerPreProcess(
        std::shared_ptr<fwk::iface::TensorIface> inputTensor,
        size_t cameraWidth,
        size_t cameraHeight) :
        m_inputTensor{inputTensor},
        m_cameraWidth{cameraWidth},
        m_cameraHeight{cameraHeight}
    {}

    void RhythmFormerPreProcess::CenterCropResize(const uint8_t* src, float* dst,
                                           size_t cropSide, size_t cropX, size_t cropY)
    {
        BilinearResizeSquare(
            src, m_cameraWidth, m_cameraHeight, cropSide, cropX, cropY, dst, kRhythmFormerImgSize);
    }

    bool RhythmFormerPreProcess::IsValidFrame(const uint8_t* rgb, size_t pixelCount)
    {
        /* Single pass over raw uint8 RGB888 data -- cheap enough to run every
         * frame at 30fps on the M55. Catches black screen / lens cap / severe
         * underexposure before diff-normalize can turn near-zero pixels into
         * amplified noise. */
        double sum = 0.0;
        double sumSq = 0.0;
        for (size_t i = 0; i < pixelCount; ++i) {
            const double v = static_cast<double>(rgb[i]);
            sum += v;
            sumSq += v * v;
        }
        const double mean = sum / static_cast<double>(pixelCount);
        double var = (sumSq / static_cast<double>(pixelCount)) - (mean * mean);
        if (var < 0.0) {
            var = 0.0;
        }
        const double stdDev = std::sqrt(var);

        /* Tracked separately from the combined pass/fail below so the
         * use-case handler can report "low lighting" specifically, distinct
         * from a bright-but-degenerate frame (e.g. a flat wall) that fails
         * on stdDev alone. */
        m_lastFrameLowLight = (mean < static_cast<double>(kMinFrameMean));

        return (mean >= static_cast<double>(kMinFrameMean)) &&
               (stdDev >= static_cast<double>(kMinFrameStd));
    }

    void RhythmFormerPreProcess::BuildChunkTensor()
    {
        /* The chunk spans frame indices [ringWriteIdx - 10 .. ringWriteIdx]
         * mod 11. That is: the 11 most-recently written frames, in the
         * chronological order they arrived. Compute the starting index so the
         * oldest frame is mapped to t=0 and the newest to t=10 (next_frame). */
        const size_t baseIdx = (m_ringWriteIdx + (kRhythmFormerFrameDepth + 1) - (kRhythmFormerFrameDepth + 1)) % (kRhythmFormerFrameDepth + 1);

        /* CenterCropResize() leaves m_frameRing holding raw 0-255-range
         * bilinear-interpolated pixel values, so normalize to [0,1] here. */
        constexpr float kPixelNorm = 1.0f / 255.0f;

        /* Quantize straight into the input tensor, reading the actual
         * deployed model's quantization params off the tensor at runtime
         * rather than hardcoding them -- stays correct if the model is
         * ever re-quantized/re-exported. */
        const auto quantParams = m_inputTensor->GetQuantParams();
        const float invScale   = 1.0f / quantParams.scale;
        const float qOffset    = static_cast<float>(quantParams.offset);
        int8_t* dst            = m_inputTensor->GetData<int8_t>();

        /* For chunk index t (0..9): diff = norm(frame[t+1]) - norm(frame[t]),
         * appearance = norm(frame[t+1]) -- the appearance channel is the
         * newer of the pair. No per-chunk mean/std standardization or
         * clipping: the raw [0,1]-normalized values are quantized directly. */
        for (size_t t = 0; t < kRhythmFormerFrameDepth; ++t) {
            const size_t curRingIdx = (baseIdx + t) % (kRhythmFormerFrameDepth + 1);
            const size_t nxtRingIdx = (baseIdx + t + 1) % (kRhythmFormerFrameDepth + 1);

            const float* cur = m_frameRing[curRingIdx].data();
            const float* nxt = m_frameRing[nxtRingIdx].data();
            int8_t* dstFrame = dst + t * kRhythmFormerImgSize * kRhythmFormerImgSize * kRhythmFormerChannels;

            for (size_t p = 0; p < kRhythmFormerImgSize * kRhythmFormerImgSize; ++p) {
                const size_t srcIdx = p * kRhythmFormerRgbChannels;
                const size_t dstIdx = p * kRhythmFormerChannels;

                for (size_t c = 0; c < kRhythmFormerRgbChannels; ++c) {
                    const float curNorm = cur[srcIdx + c] * kPixelNorm;
                    const float nxtNorm = nxt[srcIdx + c] * kPixelNorm;

                    const float diffVal = nxtNorm - curNorm;
                    const int32_t qDiff = static_cast<int32_t>(std::round(diffVal * invScale + qOffset));
                    dstFrame[dstIdx + c] = static_cast<int8_t>(std::max<int32_t>(INT8_MIN, std::min<int32_t>(INT8_MAX, qDiff)));

                    const float appVal = nxtNorm;
                    const int32_t qApp = static_cast<int32_t>(std::round(appVal * invScale + qOffset));
                    dstFrame[dstIdx + kRhythmFormerRgbChannels + c] = static_cast<int8_t>(std::max<int32_t>(INT8_MIN, std::min<int32_t>(INT8_MAX, qApp)));
                }
            }
        }
    }

    bool RhythmFormerPreProcess::DoPreProcess(const void* input, size_t inputSize)
    {
        m_chunkReady = false;

        if (input == nullptr) {
            printf_err("RhythmFormerPreProcess: null input\n");
            return false;
        }

        const auto* rgb = static_cast<const uint8_t*>(input);
        const size_t expectedSize =
            m_cameraWidth * m_cameraHeight * kRhythmFormerRgbChannels;
        if (inputSize < expectedSize) {
            printf_err("RhythmFormerPreProcess: input too small (%zu < %zu)\n", inputSize, expectedSize);
            return false;
        }

        if (!IsValidFrame(rgb, expectedSize)) {
            /* Invalid frame (too dark, lens cap, etc). Increment the streak
             * counter and only hard-reset chunk accumulation once
             * kMaxInvalidFrameStreak consecutive bad frames have occurred,
             * so a single transient (auto-exposure blip, blink) doesn't
             * discard the buffer. */
            ++m_invalidStreak;
            if (m_invalidStreak >= kMaxInvalidFrameStreak) {
                debug("RhythmFormerPreProcess: %zu consecutive invalid frames (mean/std below threshold), "
                      "resetting chunk accumulation.\n", m_invalidStreak);
                Reset();
            } else {
                /* Within the grace period: skip this frame but leave the
                 * ring buffer and counters alone so accumulation resumes
                 * normally on the next valid frame. */
                debug("RhythmFormerPreProcess: invalid frame (streak=%zu), skipping (threshold=%zu)\n",
                      m_invalidStreak, kMaxInvalidFrameStreak);
            }
            return true;
        }
        /* Good frame: reset the invalid-streak counter. */
        m_invalidStreak = 0;

        /* Fixed centered ROI: no subject-detection step, every frame is
         * cropped to the same centered square (kCenterCropFrac of the
         * shorter camera dimension) before being resized into the ring. */
        const size_t minCamSide = std::min(m_cameraWidth, m_cameraHeight);
        const size_t cropSide = static_cast<size_t>(kCenterCropFrac * static_cast<float>(minCamSide));
        const size_t cropX = (m_cameraWidth  - cropSide) / 2;
        const size_t cropY = (m_cameraHeight - cropSide) / 2;

        CenterCropResize(rgb, m_frameRing[m_ringWriteIdx].data(), cropSide, cropX, cropY);
        m_ringWriteIdx = (m_ringWriteIdx + 1) % (kRhythmFormerFrameDepth + 1);
        ++m_frameCount;

        /* Need kRhythmFormerFrameDepth + 1 frames before the first chunk (extra for diff). */
        if (m_frameCount < kRhythmFormerFrameDepth + 1) {
            debug("RhythmFormerPreProcess: buffering frame %zu/%zu\n",
                  m_frameCount,
                  kRhythmFormerFrameDepth + 1);
            return true;
        }

        /* Stride-1 sliding window: run inference every frame once the
         * ring is full, producing exactly one new BVP sample per pass
         * (kBvpNewSamplesPerInference), so BVP sample spacing matches the
         * assumed fps. */
        const size_t framesSinceFull = m_frameCount - (kRhythmFormerFrameDepth + 1);
        if ((framesSinceFull % kRhythmFormerStride) != 0) {
            return true;
        }

        BuildChunkTensor();
        m_chunkReady = true;
        debug("RhythmFormerPreProcess: chunk ready at frame %zu (stride=%zu, overlap=%zu/%zu)\n",
              m_frameCount,
              static_cast<size_t>(kRhythmFormerStride),
              static_cast<size_t>(kRhythmFormerFrameDepth - kRhythmFormerStride),
              static_cast<size_t>(kRhythmFormerFrameDepth));
        return true;
    }

    /* -------------------------------------------------------------------- */
    /* RhythmFormerPostProcess implementation                                       */
    /* -------------------------------------------------------------------- */

    RhythmFormerPostProcess::RhythmFormerPostProcess(
        std::shared_ptr<fwk::iface::TensorIface> outputTensor,
        RhythmFormerResult& result,
        float assumedFps) :
        m_outputTensor{outputTensor},
        m_result{result},
        m_assumedFps{assumedFps}
    {
        ResetAll();
    }

    void RhythmFormerPostProcess::ResetAll()
    {
        m_bvpHistoryCount = 0;
        m_avgBatchSize = 0;
        m_avgBatchWriteIdx = 0;
        m_smoothedBpm = 0.0f;
        m_avgConfidence = 0.0f;
        m_msSinceLastDisplayUpdate = 0;
        m_publishedHr = 0.0f;
        m_result.hrValid = false;
        m_result.hrBpm = 0.0f;
        m_result.rawHrBpm = 0.0f;
        m_result.avgConfidence = 0.0f;
    }

    void RhythmFormerPostProcess::TickMs(uint32_t elapsedMs)
    {
        /* Drives the display-refresh throttle only; the rolling average
         * itself updates every cycle in UpdateHeartRateEstimate()
         * regardless of this timer. */
        m_msSinceLastDisplayUpdate += elapsedMs;
    }

    void RhythmFormerPostProcess::DesignButterBandpass(float fps, float lowHz, float highHz,
                                                ButterSos* sectionsOut, size_t& numSectionsOut)
    {
        /* Always 2 SOS sections for a 2nd-order Butterworth bandpass.
         * Gain is inherent to each section's numerator (see
         * ComputeButter2BpSosCoefs) -- no separate scale factor needed. */
        numSectionsOut = 2;
        ComputeButter2BpSosCoefs(fps, lowHz, highHz, sectionsOut[0], sectionsOut[1]);
    }

    void RhythmFormerPostProcess::FilterSosPass(const ButterSos& sec, float* buf, size_t len,
                                         bool reverse)
    {
        /* Direct-Form-II Transposed, one second-order section, one pass.
         * State is two delay elements per section. Filtering either forward
         * or reverse over the same buffer; reverse = filtfilt second pass. */
        float z1 = 0.0f;
        float z2 = 0.0f;
        if (reverse) {
            for (int64_t i = static_cast<int64_t>(len) - 1; i >= 0; --i) {
                const float x = buf[i];
                const float y = sec.b0 * x + z1;
                z1 = sec.b1 * x - sec.a1 * y + z2;
                z2 = sec.b2 * x - sec.a2 * y;
                buf[i] = y;
            }
        } else {
            for (size_t i = 0; i < len; ++i) {
                const float x = buf[i];
                const float y = sec.b0 * x + z1;
                z1 = sec.b1 * x - sec.a1 * y + z2;
                z2 = sec.b2 * x - sec.a2 * y;
                buf[i] = y;
            }
        }
    }

    void RhythmFormerPostProcess::FiltFiltSos(const ButterSos* sections, size_t numSections,
                                       float* workBuf, size_t len)
    {
        if (len < 3 || sections == nullptr) {
            return;
        }
        /* Forward pass, one section at a time (cascade). */
        for (size_t s = 0; s < numSections; ++s) {
            FilterSosPass(sections[s], workBuf, len, /*reverse=*/false);
        }
        /* Reverse pass (zero-phase correction), same cascade order reversed.
         * (Applying sections in the same order, but on reversed data, gives
         * overall zero-phase; section order inside the reverse pass is a
         * detail since SOS commute within numeric precision for a cascade.) */
        for (size_t s = 0; s < numSections; ++s) {
            FilterSosPass(sections[s], workBuf, len, /*reverse=*/true);
        }
    }

    void RhythmFormerPostProcess::ApplyHannWindow(float* buf, size_t len)
    {
        const float twoPiOverN = 2.0f * static_cast<float>(M_PI) /
                                 std::max<uint32_t>(1u, static_cast<uint32_t>(len - 1));
        for (size_t n = 0; n < len; ++n) {
            const float w = 0.5f * (1.0f - std::cos(twoPiOverN * static_cast<float>(n)));
            buf[n] *= w;
        }
    }

    float RhythmFormerPostProcess::EstimateHrFromBuffer(const float* signal,
                                                 size_t len,
                                                 float fps,
                                                 float lowHz,
                                                 float highHz,
                                                 float& outConfidence)
    {
        outConfidence = 0.f;
        /* UpdateHeartRateEstimate() already gates on kMinBufferSamples
         * before calling this, so len < 30 shouldn't happen in practice;
         * kept as a hard safety floor for direct/test callers. */
        if (len < 30 || fps <= 0.f) {
            return 0.f;
        }

        /* Step 1: detrend and standardize (zero mean, unit variance). */
        float mean = 0.f;
        float invStd = 1.0f;
        ComputeMeanAndInvStd(signal, len, mean, invStd);

        static float work[kBvpHistoryLen];
        static float workFiltered[kBvpHistoryLen];
        for (size_t i = 0; i < len; ++i) {
            work[i] = (signal[i] - mean) * invStd;
        }

        /* Step 2: 2nd-order Butterworth bandpass, filtfilt (zero-phase). */
        ButterSos sections[2];
        size_t numSec = 0;
        DesignButterBandpass(fps, lowHz, highHz, sections, numSec);
        std::memcpy(workFiltered, work, len * sizeof(float));
        FiltFiltSos(sections, numSec, workFiltered, len);

        /* Step 3: Hann window. */
        ApplyHannWindow(workFiltered, len);

        /* Step 4: in-band spectral scan. A full zero-padded FFT isn't
         * viable per-cycle on the M55, so this scans the band with a
         * per-bin Goertzel filter at the buffer's native frequency
         * resolution (fps/len) instead -- coarser bins, but the same
         * peak-search and peak/mean power ratio that drive the reported HR
         * and confidence gate. */
        const size_t nBins   = len / 2;
        float maxMag         = 0.f;
        float peakFreq       = 0.f;
        size_t peakBin       = 0;
        const float freqStep = fps / static_cast<float>(len);

        static float binPower[kBvpHistoryLen / 2];
        size_t nBinsRecorded = 0;
        double totalPower    = 0.0;

        for (size_t k = 1; k < nBins; ++k) {
            const float freq = static_cast<float>(k) * freqStep;
            if (freq < lowHz || freq > highHz) {
                continue;
            }

            const float omega = 2.0f * static_cast<float>(M_PI) * static_cast<float>(k) / static_cast<float>(len);
            const float coeff = 2.0f * std::cos(omega);

            float s0 = 0.0f;
            float s1 = 0.0f;
            float s2 = 0.0f;

            for (size_t n = 0; n < len; ++n) {
                s0 = workFiltered[n] + coeff * s1 - s2;
                s2 = s1;
                s1 = s0;
            }

            const float power = s1 * s1 + s2 * s2 - s1 * s2 * coeff;
            if (nBinsRecorded < (kBvpHistoryLen / 2)) {
                binPower[nBinsRecorded++] = power;
            }
            totalPower += static_cast<double>(power);

            if (power > maxMag) {
                maxMag   = power;
                peakFreq = freq;
                peakBin  = nBinsRecorded - 1;
            }
        }
        (void)peakBin;

        if (peakFreq <= 0.f || nBinsRecorded == 0) {
            return 0.f;
        }

        /* Step 5: confidence = peak-to-mean in-band power ratio, not a
         * peak-window/total-power ratio -- a flat (noisy) spectrum scores
         * snr~1 (0% confidence) regardless of in-band bin count, while a
         * sharp single-bin peak among flat bins scores high. */
        const double meanPower = (totalPower / static_cast<double>(nBinsRecorded)) + 1e-6;
        const float snr = static_cast<float>(static_cast<double>(maxMag) / meanPower);
        const float confFrac = (snr - kSnrConfidenceFloor) / kSnrConfidenceSpan;
        outConfidence = std::max(0.0f, std::min(1.0f, confFrac));

        return peakFreq * 60.f;
    }


    void RhythmFormerPostProcess::UpdateHeartRateEstimate()
    {
        if (m_bvpHistoryCount < kMinBufferSamples) {
            m_result.hrValid = false;
            return;
        }

        float confidence = 0.f;
        const float hr = EstimateHrFromBuffer(
            m_bvpHistory.data(), m_bvpHistoryCount, m_assumedFps,
            kHrLowHz, kHrHighHz, confidence);

        m_result.hrConfidence = confidence;
        m_result.rawHrBpm = (hr > 0.f) ? hr : 0.0f;

        /* Confidence-gated rolling average: only fold hr into the batch
         * when its confidence clears kMinHrConfidence. */
        if (hr > 0.f && confidence >= kMinHrConfidence) {
            m_avgBatch[m_avgBatchWriteIdx] = hr;
            m_avgConfidenceBatch[m_avgBatchWriteIdx] = confidence;
            m_avgBatchWriteIdx = (m_avgBatchWriteIdx + 1) % kAvgWindowCycles;
            if (m_avgBatchSize < kAvgWindowCycles) {
                ++m_avgBatchSize;
            }
        }

        if (m_avgBatchSize > 0) {
            double sum = 0.0;
            double confSum = 0.0;
            for (size_t i = 0; i < m_avgBatchSize; ++i) {
                sum += static_cast<double>(m_avgBatch[i]);
                confSum += static_cast<double>(m_avgConfidenceBatch[i]);
            }
            m_smoothedBpm   = static_cast<float>(sum / static_cast<double>(m_avgBatchSize));
            m_avgConfidence = static_cast<float>(confSum / static_cast<double>(m_avgBatchSize));
        }
        /* else: keep the last m_smoothedBpm/m_avgConfidence -- a single
         * low-confidence cycle shouldn't blank out a good reading. */

        /* Throttled display publish: BPM and its average confidence
         * refresh together so the two numbers never describe different
         * cycles. */
        if (m_msSinceLastDisplayUpdate >= kDisplayUpdateIntervalMs) {
            m_publishedHr = m_smoothedBpm;
            m_result.avgConfidence = m_avgConfidence;
            m_msSinceLastDisplayUpdate = 0;
        }

        m_result.hrBpm   = m_publishedHr;
        m_result.hrValid = true; /* past the warm-up/calibration window. */
    }

    bool RhythmFormerPostProcess::DoPostProcess()
    {
        const float* outData = m_outputTensor->GetData<float>();
        if (outData == nullptr) {
            printf_err("RhythmFormerPostProcess: null output tensor\n");
            return false;
        }

        /* Output shape: (T, 1) float32. Save the raw vector for diagnostics. */
        for (size_t t = 0; t < kRhythmFormerFrameDepth; ++t) {
            m_result.bvpRaw[t] = outData[t];
        }
        ++m_result.chunkIndex;

        /* Append only the genuinely new BVP samples: with a stride-1
         * sliding window, 9/10 of the output vector overlaps frame windows
         * already seen in previous iterations, so only the last sample is
         * new. Appending all 10 would duplicate 9 stale samples and smear
         * the cardiac peak. */
        static_assert(kRhythmFormerStride >= 1, "Stride must be at least 1");
        static_assert(kBvpNewSamplesPerInference <= kRhythmFormerFrameDepth,
                      "Cannot append more samples than the model produces per inference");
        const size_t firstNewIdx = kRhythmFormerFrameDepth - kBvpNewSamplesPerInference;
        for (size_t i = 0; i < kBvpNewSamplesPerInference; ++i) {
            const size_t srcIdx = firstNewIdx + i;
            const float sample = m_result.bvpRaw[srcIdx];
            if (m_bvpHistoryCount < kBvpHistoryLen) {
                m_bvpHistory[m_bvpHistoryCount++] = sample;
            }
            /* If already at kBvpHistoryLen (== kSessionMaxPoints) this
             * sample is simply dropped for the rest of this cycle -- the
             * session-rollover check below fires first in practice since it
             * runs every cycle once the count reaches the cap. */
        }

        UpdateHeartRateEstimate();

        /* Session rollover: once the BVP history reaches kSessionMaxPoints
         * the whole buffer is cleared and accumulation restarts from 0. The
         * display averaging batch is untouched by this, so the very next
         * cycle's raw_bpm=0/confidence=0 estimate
         * (buffer too short) simply fails the confidence gate above and the
         * displayed average holds steady through the reset. */
        if (m_bvpHistoryCount >= kSessionMaxPoints) {
            m_bvpHistoryCount = 0;
        }

        /* Condensed UART print. Print raw (unaveraged) hr and confidence
         * alongside the throttled displayed number so operators can tune
         * lighting / distance and watch confidence climb without having to
         * wait for the next 5s display refresh. */
        const bool displayValid = m_result.hrValid && m_result.hrBpm > 0.0f;
        if (displayValid) {
            info("RhythmFormer chunk %zu: Display HR=%.1f bpm (avg of last %zu confident cycles, "
                 "refreshed every %ums) | raw HR=%.1f conf=%.2f | BVP new[last]=%+.2f\n",
                 m_result.chunkIndex,
                 static_cast<double>(m_result.hrBpm),
                 m_avgBatchSize,
                 static_cast<unsigned>(kDisplayUpdateIntervalMs),
                 static_cast<double>(m_result.rawHrBpm),
                 static_cast<double>(m_result.hrConfidence),
                 static_cast<double>(m_result.bvpRaw[kRhythmFormerFrameDepth - 1]));
        } else {
            info("RhythmFormer chunk %zu: HR buffering or low-confidence "
                 "(%zu/%zu BVP samples, raw=%.1f conf=%.2f, thr=%.2f)\n",
                 m_result.chunkIndex,
                 m_bvpHistoryCount,
                 static_cast<size_t>(kMinBufferSamples),
                 static_cast<double>(m_result.rawHrBpm),
                 static_cast<double>(m_result.hrConfidence),
                 static_cast<double>(kMinHrConfidence));
        }

        return true;
    }

    /* =========================================================================
     * FaceDetectionState — face-detection scheduling state machine
     * ========================================================================= */

    bool FaceDetectionState::ShouldRunDetection() const
    {
        /* Never run detection during a recalibration hold, and never run it
         * during the skip window that follows a positive detection. */
        return !recalibrating && (skipFramesRemaining == 0U);
    }

    void FaceDetectionState::OnFaceDetected()
    {
        noFaceStreak        = 0U;
        skipFramesRemaining = kFaceDetectSkipFrames;
    }

    bool FaceDetectionState::OnNoFaceDetected()
    {
        ++noFaceStreak;
        if (noFaceStreak >= kFaceDetectNoFaceThreshold) {
            return true;  /* caller must call BeginRecalibration() */
        }
        return false;
    }

    bool FaceDetectionState::OnMotionDetected()
    {
        ++motionStreak;
        if (motionStreak >= kFaceDetectMotionThreshold) {
            return true;  /* caller must call BeginRecalibration() */
        }
        return false;
    }

    void FaceDetectionState::OnNoMotionDetected()
    {
        motionStreak = 0U;
    }

    void FaceDetectionState::BeginRecalibration()
    {
        recalibrating      = true;
        recalibMsRemaining = kFaceDetectRecalibMs;
        noFaceStreak       = 0U;
        motionStreak       = 0U;
        skipFramesRemaining = 0U;
    }

    bool FaceDetectionState::TickMs(uint32_t elapsedMs)
    {
        /* Decrement skip counter each frame regardless of recalibration state. */
        if (skipFramesRemaining > 0U) {
            --skipFramesRemaining;
        }

        if (!recalibrating) {
            return false;
        }

        if (elapsedMs >= recalibMsRemaining) {
            /* Hold period has expired — exit recalibration mode. */
            recalibMsRemaining = 0U;
            recalibrating      = false;
            return true;  /* one-shot completion event */
        }

        recalibMsRemaining -= elapsedMs;
        return false;
    }

    void FaceDetectionState::Reset()
    {
        recalibrating       = false;
        skipFramesRemaining = 0U;
        noFaceStreak        = 0U;
        motionStreak        = 0U;
        recalibMsRemaining  = 0U;
    }

} /* namespace app */
} /* namespace arm */