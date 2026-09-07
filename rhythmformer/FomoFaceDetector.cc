#include "mlek/use_case/rhythmformer/FomoFaceDetector.hpp"
#include "mlek/log/log_macros.h"

#include <cstdint>

namespace arm {
namespace app {


/* ---------------------------------------------------------------------------
 * PreProcess
 *
 * Nearest-neighbour resize (srcWidth × srcHeight) → (72 × 72) combined with
 * the FOMO model's required quantisation in a single pass:
 *
 *   int8_val = (int16_t)uint8_pixel - 128
 *
 * The model was exported with scale=1/255, zero_point=-128, so for uint8
 * pixel p: q = round(p/255 / (1/255)) + (-128) = p - 128. Because
 * p ∈ [0, 255], (p - 128) ∈ [-128, 127] — always in int8 range, so no
 * clamping is needed.
 *
 * Results are written directly into the model's input tensor buffer via
 * GetData<int8_t>() to avoid an extra memcpy.
 * --------------------------------------------------------------------------*/
void FomoFaceDetector::PreProcess(const uint8_t*     rgb888,
                                  size_t             srcWidth,
                                  size_t             srcHeight,
                                  fwk::iface::Model& model)
{
    /* Write directly into the model's own input tensor buffer. */
    int8_t* dst = model.GetInputTensor(0)->GetData<int8_t>();

    for (int dy = 0; dy < kInputHeight; ++dy) {
        /* Nearest-neighbour row mapping: dst row dy → src row sy. */
        const size_t sy =
            (static_cast<size_t>(dy) * srcHeight) / static_cast<size_t>(kInputHeight);
        const uint8_t* srcRow = rgb888 + sy * srcWidth * 3U;

        for (int dx = 0; dx < kInputWidth; ++dx) {
            /* Nearest-neighbour column mapping: dst col dx → src col sx. */
            const size_t sx =
                (static_cast<size_t>(dx) * srcWidth) / static_cast<size_t>(kInputWidth);
            const uint8_t* px = srcRow + sx * 3U;

            const size_t dstBase =
                (static_cast<size_t>(dy) * static_cast<size_t>(kInputWidth) +
                 static_cast<size_t>(dx)) * static_cast<size_t>(kInputChannels);

            /* Quantise: int8_val = (int16_t)pixel - 128.
             * No clamp needed: uint8 ∈ [0,255] → result ∈ [-128,127]. */
            for (int c = 0; c < kInputChannels; ++c) {
                dst[dstBase + static_cast<size_t>(c)] =
                    static_cast<int8_t>(static_cast<int16_t>(px[c]) - 128);
            }
        }
    }
}


/* ---------------------------------------------------------------------------
 * DetectFace
 *
 * Runs one full FOMO inference pass and returns true if any cell in the 9×9
 * output grid has a dequantised face probability ≥ confThreshold.
 *
 * Output tensor layout: [1, kGridSize, kGridSize, 2] int8
 *   index = (gy * kGridSize + gx) * 2 + channel
 *   channel 0 = background, channel 1 = face (kFaceClassIdx)
 *
 * Dequantisation uses the tensor's own quantisation parameters so the result
 * is always consistent with whatever Vela embedded in the flatbuffer.
 * --------------------------------------------------------------------------*/
bool FomoFaceDetector::DetectFace(fwk::iface::Model& model,
                                  const uint8_t*     rgb888,
                                  size_t             width,
                                  size_t             height,
                                  float              confThreshold)
{
    if (rgb888 == nullptr || width == 0U || height == 0U) {
        return false;
    }

    if (!model.IsInited()) {
        printf_err("FomoFaceDetector: model is not initialised\n");
        return false;
    }

    PreProcess(rgb888, width, height, model);

    /* One-time diagnostic on first call: log input tensor stats to confirm
     * that PreProcess is writing real camera pixel data (not zeros). */
    {
        static bool s_dumpedOnce = false;
        if (!s_dumpedOnce) {
            auto inT = model.GetInputTensor(0);
            const int8_t* p = inT ? inT->GetData<int8_t>() : nullptr;
            if (p != nullptr) {
                size_t nz = 0;
                const size_t n = static_cast<size_t>(kInputWidth) *
                                 static_cast<size_t>(kInputHeight) *
                                 static_cast<size_t>(kInputChannels);
                for (size_t i = 0; i < n; ++i) {
                    if (p[i] != 0) { ++nz; }
                }
                info("[FOMO_DIAG] input_bytes=%zu nonzero=%zu "
                     "qp.scale=%.6f qp.zp=%d samples=[%d,%d,%d,%d,%d,%d]\n",
                     n, nz,
                     static_cast<double>(inT->GetQuantParams().scale),
                     static_cast<int>(inT->GetQuantParams().offset),
                     static_cast<int>(p[0]),    static_cast<int>(p[1]),
                     static_cast<int>(p[2]),    static_cast<int>(p[15549]),
                     static_cast<int>(p[15550]), static_cast<int>(p[15551]));
            }
            s_dumpedOnce = true;
        }
    }

    if (!model.RunInference()) {
        printf_err("FomoFaceDetector: inference failed\n");
        return false;
    }

    /* Output tensor: shape [1, kGridSize, kGridSize, 2], int8.
     * Dequantise using the tensor's own quantisation parameters. */
    auto outT = model.GetOutputTensor(0);
    if (!outT) {
        printf_err("FomoFaceDetector: null output tensor\n");
        return false;
    }

    const int8_t* rawOut   = outT->GetData<int8_t>();
    const fwk::iface::QuantParams qp = outT->GetQuantParams();
    const float scale      = qp.scale;
    const int32_t zp       = qp.offset;

    float maxFaceProb = -1.0f;
    size_t nAbove     = 0;

    const int kCells = kGridSize * kGridSize;   /* 81 cells */
    for (int cell = 0; cell < kCells; ++cell) {
        /* Layout: [..., 2] → face channel is at index cell*2 + kFaceClassIdx. */
        const int8_t qFace = rawOut[cell * 2 + kFaceClassIdx];
        const float faceProb =
            (static_cast<float>(qFace) - static_cast<float>(zp)) * scale;

        if (faceProb > maxFaceProb) {
            maxFaceProb = faceProb;
        }
        if (faceProb >= confThreshold) {
            ++nAbove;
        }
    }

    const bool detected = nAbove > 0U;

    info("[FOMO_FACE] detected=%d n_above=%zu max_prob=%.4f thr=%.2f\n",
         detected ? 1 : 0,
         nAbove,
         static_cast<double>(maxFaceProb),
         static_cast<double>(confThreshold));

    return detected;
}

} /* namespace app */
} /* namespace arm */
