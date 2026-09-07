#ifndef RHYTHMFORMER_PROCESSING_HPP
#define RHYTHMFORMER_PROCESSING_HPP

#include "mlek/common/BaseProcessing.hpp"
#include "mlek/fwk/iface/Model.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#define RHYTHMFORMER_DRAM_ATTR __attribute__((section(".bss.NoInit.activation_buf_dram"), aligned(16)))
#else
#define RHYTHMFORMER_DRAM_ATTR
#endif

namespace arm {
namespace app {

    /** Assumed camera frame rate, used only to convert the seconds-based
     *  constants below into frame counts at compile time. Must match the fps
     *  passed to RhythmFormerPostProcess's constructor. UseCaseHandler.cc
     *  re-declares the same default independently since this header is
     *  compiled as a separate translation unit. Overridable via
     *  -Dalif_RhythmFormer_RHYTHMFORMER_ASSUMED_FPS=<float>. */
#ifndef RHYTHMFORMER_ASSUMED_FPS
#define RHYTHMFORMER_ASSUMED_FPS 30.0f
#endif

    /** RhythmFormer model constants (fd10 / 72x72 efficient int8). */
    static constexpr size_t kRhythmFormerFrameDepth = 10;
    static constexpr size_t kRhythmFormerImgSize    = 72;
    static constexpr size_t kRhythmFormerChannels   = 6;
    static constexpr size_t kRhythmFormerRgbChannels = 3;

    /** Stride between inference passes, in frames. stride=1 is a true
     *  sliding window: every camera frame triggers one inference, and only
     *  the last output sample is genuinely new (see
     *  kBvpNewSamplesPerInference), keeping BVP sample spacing consistent
     *  with the assumed fps. */
    static constexpr size_t kRhythmFormerStride = 1;

    /** New BVP samples appended per inference pass. Always 1 for stride=1:
     *  only the newest output sample wasn't already produced by a previous
     *  overlapping window. */
    static constexpr size_t kBvpNewSamplesPerInference = 1;

    /** Frame-validity guard thresholds (0-255, raw RGB888). Rejects a
     *  black/near-black frame (lens cap, severe underexposure) before
     *  diff-normalize can amplify sensor noise into a false signal. */
    static constexpr float kMinFrameMean = 10.0f;
    static constexpr float kMinFrameStd  = 5.0f;

    /** Consecutive invalid frames tolerated before a hard reset. Absorbs a
     *  brief blink or auto-exposure transient without discarding the
     *  buffer. */
    static constexpr size_t kMaxInvalidFrameStreak = 5;

    /** Centered crop size, as a fraction of min(cameraWidth, cameraHeight).
     *  No subject detection — every frame uses the same centered square. */
    static constexpr float kCenterCropFrac = 0.40f;

    /** Minimum confidence (0-1) for a per-chunk HR estimate to be folded
     *  into the displayed rolling average; below this it's still computed
     *  for diagnostics but excluded. Confidence is the peak-to-mean in-band
     *  spectral power ratio (see EstimateHrFromBuffer). */
    static constexpr float kMinHrConfidence = 0.5f;

    /** Maps in-band peak/mean power ratio (snr) to a 0-1 confidence score:
     *  conf = clip((snr - floor) / span, 0, 1). snr at or below floor gives
     *  0% confidence; snr >= floor+span saturates at 100%. */
    static constexpr float kSnrConfidenceFloor = 1.0f;
    static constexpr float kSnrConfidenceSpan  = 1.5f;

    /** BVP waveform history capacity, expressed in seconds
     *  (RHYTHMFORMER_SESSION_SECONDS) rather than a raw frame count. Once
     *  full, the buffer is cleared and a new session starts. Must stay
     *  comfortably larger than the calibration window so the session never
     *  rolls over before hrValid can become true. Overridable via
     *  -Dalif_RhythmFormer_RHYTHMFORMER_SESSION_SECONDS=<float>. */
#ifndef RHYTHMFORMER_SESSION_SECONDS
#define RHYTHMFORMER_SESSION_SECONDS 30.0f
#endif
    static constexpr size_t kBvpHistoryLen =
        static_cast<size_t>((RHYTHMFORMER_SESSION_SECONDS) * (RHYTHMFORMER_ASSUMED_FPS) + 0.5f);
    static constexpr size_t kSessionMaxPoints = kBvpHistoryLen;

    /** Warm-up gate: no HR is computed until this many BVP points have been
     *  collected in the current session. Expressed in seconds so the
     *  calibration duration is fps-independent. Overridable via
     *  -Dalif_RhythmFormer_RHYTHMFORMER_CALIBRATION_SECONDS=<float>; must
     *  stay well below kBvpHistoryLen (see static_assert below). */
#ifndef RHYTHMFORMER_CALIBRATION_SECONDS
#define RHYTHMFORMER_CALIBRATION_SECONDS 15.0f
#endif
    static constexpr float  kCalibrationSeconds = RHYTHMFORMER_CALIBRATION_SECONDS;
    static constexpr size_t kMinBufferSamples =
        static_cast<size_t>((RHYTHMFORMER_CALIBRATION_SECONDS) * (RHYTHMFORMER_ASSUMED_FPS) + 0.5f);

    static_assert(kMinBufferSamples < kBvpHistoryLen,
                  "RHYTHMFORMER_CALIBRATION_SECONDS must fit within RHYTHMFORMER_SESSION_SECONDS "
                  "with room to spare, or the session will roll over before "
                  "calibration ever completes");

    /** Heart-rate band, in Hz (45-120 bpm). */
    static constexpr float kHrLowHz  = 0.75f;
    static constexpr float kHrHighHz = 2.00f;

    /** Rolling-average window for the displayed HR, in inference cycles.
     *  Each cycle contributes at most one confidence-gated sample; the
     *  display value is the mean of the last kAvgWindowCycles of them. */
    static constexpr size_t kAvgWindowCycles = 16;

    /** Minimum interval between displayed-HR refreshes. The underlying
     *  rolling average still updates every cycle; only the published value
     *  is throttled. */
    static constexpr uint32_t kDisplayUpdateIntervalMs = 5000;

    /**
     * @brief   Pre-processing for RhythmFormer: accumulates camera frames, builds the
     *          (T, H, W, 6) motion+appearance tensor expected by the model.
     */
    class RhythmFormerPreProcess : public BasePreProcess {

    public:
        explicit RhythmFormerPreProcess(
            const std::shared_ptr<fwk::iface::TensorIface> inputTensor,
            size_t cameraWidth,
            size_t cameraHeight);

        /**
         * @brief   Ingest one RGB888 camera frame. When a full chunk is ready
         *          (at least kRhythmFormerFrameDepth+1 frames buffered AND current
         *          frame is on the configured stride boundary), populates the
         *          input tensor and returns true.
         */
        bool DoPreProcess(const void* input, size_t inputSize) override;

        /** Returns true when the latest DoPreProcess filled a chunk tensor. */
        bool IsChunkReady() const
        {
            return m_chunkReady;
        }

        /** Reset all accumulator state: ring buffer, frame counter, invalid
         *  streak counter. Equivalent to a full buffer clear-and-restart. */
        void Reset()
        {
            m_frameCount  = 0;
            m_chunkReady  = false;
            m_ringWriteIdx = 0;
            m_invalidStreak = 0;
            m_lastFrameLowLight = false;
        }

        /** True if the most recently processed frame's mean brightness fell
         *  below kMinFrameMean -- regardless of whether IsValidFrame's
         *  combined mean+std check ultimately accepted or rejected the
         *  frame. Exposed so the use-case handler can report a distinct
         *  "low lighting" state to the UI instead of a generic "invalid
         *  frame" catch-all (which also covers e.g. a lens-capped/blank
         *  frame that is bright but has near-zero variance). */
        bool WasLastFrameLowLight() const { return m_lastFrameLowLight; }

    private:
        std::shared_ptr<fwk::iface::TensorIface> m_inputTensor;
        size_t m_cameraWidth;
        size_t m_cameraHeight;
        size_t m_frameCount = 0;

        using FrameRing = std::array<std::array<float, kRhythmFormerImgSize * kRhythmFormerImgSize * kRhythmFormerRgbChannels>,
                                     kRhythmFormerFrameDepth + 1>;
        static FrameRing m_frameRing RHYTHMFORMER_DRAM_ATTR;
        size_t m_ringWriteIdx = 0;
        bool m_chunkReady     = false;
        size_t m_invalidStreak = 0;
        bool m_lastFrameLowLight = false;

        void CenterCropResize(const uint8_t* src, float* dst,
                              size_t cropSide, size_t cropX, size_t cropY);
        void BuildChunkTensor();

        /** Cheap brightness/variance check on the raw camera frame, run
         *  before it's accepted into the frame ring. Returns false on a
         *  black/degenerate frame (lens cap, dark room, camera fault). Also
         *  updates m_lastFrameLowLight -- hence non-const, unlike before. */
        bool IsValidFrame(const uint8_t* rgb, size_t pixelCount);

    };

    /** Heart-rate estimation result from accumulated BVP samples. */
    struct RhythmFormerResult {
        float bvpRaw[kRhythmFormerFrameDepth]{};  /**< Raw BVP from last chunk inference. */
        float hrBpm       = 0.f;           /**< Current displayed HR (mean of last averaging window, or 0 if unknown). */
        float hrConfidence = 0.f;          /**< In-band peak power ratio of the latest raw estimate, 0-1. */
        bool  hrValid     = false;         /**< True when hrBpm came from a valid averaging batch. */
        size_t totalFrames = 0;            /**< Total camera frames processed so far. */
        size_t chunkIndex  = 0;            /**< Number of inference passes completed. */
        float  rawHrBpm    = 0.f;          /**< Raw (unaveraged) latest per-chunk HR estimate, or 0 if rejected. */
        float  avgConfidence = 0.f;        /**< Mean confidence (0-1) of the samples currently in the
                                             *   displayed averaging window (see kAvgWindowCycles). 0 when
                                             *   the window is empty (no accepted sample yet / subject just
                                             *   left frame), in which case hrConfidence (the latest raw
                                             *   per-cycle value) is the only signal available. */
    };

    /**
     * @brief   Post-processing: dequantizes BVP output, accumulates waveform,
     *          estimates heart rate with bandpass + Hann window + Goertzel,
     *          keeps a rolling average of confidence-gated per-chunk
     *          estimates, and throttles the published/displayed HR to refresh
     *          at most once every kDisplayUpdateIntervalMs.
     */
    class RhythmFormerPostProcess : public BasePostProcess {

    public:
        RhythmFormerPostProcess(const std::shared_ptr<fwk::iface::TensorIface> outputTensor,
                         RhythmFormerResult& result,
                         float assumedFps);

        bool DoPostProcess() override;

        /** Advance the internal wall-clock timer that throttles the
         *  *displayed* HR value. Pass the elapsed milliseconds
         *  since the last call (or since construction for the first call).
         *  Called by the use-case handler at the end of every inference pass
         *  so the throttle uses actual wall time, not frame counts. The
         *  rolling average itself (m_smoothedBpm) is not gated by this
         *  timer -- it updates every cycle. */
        void TickMs(uint32_t elapsedMs);

        /** Estimate HR from accumulated BVP buffer (call periodically).
         *  Exposed for tests; normally called internally by DoPostProcess(). */
        void UpdateHeartRateEstimate();

        /** Full reset: drop BVP history, averaging batch, clock. */
        void ResetAll();

        /** Number of valid points currently in the BVP waveform history
         *  buffer (0..kBvpHistoryLen-1; resets to 0 on session rollover at
         *  kSessionMaxPoints). Exposed so the use-case handler can log the
         *  waveform for external UI tools without the library taking on any
         *  HAL/logging dependency itself. */
        size_t GetBvpHistoryCount() const { return m_bvpHistoryCount; }

        /** Pointer to the BVP waveform history buffer. The first
         *  GetBvpHistoryCount() entries are valid; the rest is stale data
         *  from a previous session. */
        const float* GetBvpHistory() const { return m_bvpHistory.data(); }

        /** Current per-cycle rolling-average HR. Updates every inference
         *  cycle, unlike RhythmFormerResult::hrBpm which is
         *  throttled to refresh at most every kDisplayUpdateIntervalMs. */
        float GetSmoothedBpm() const { return m_smoothedBpm; }

        /** Second-order section (biquad) coefficients for SOS-based IIR
         *  filtering. Public so helper functions in the translation unit
         *  can take references to it. */
        struct ButterSos {
            float b0, b1, b2;
            float a1, a2;
        };

    private:
        std::shared_ptr<fwk::iface::TensorIface> m_outputTensor;
        RhythmFormerResult& m_result;
        float m_assumedFps;

        using BvpHistory = std::array<float, kBvpHistoryLen>;
        static BvpHistory m_bvpHistory RHYTHMFORMER_DRAM_ATTR;
        static size_t m_bvpHistoryCount;

        /** Ring buffer of the last kAvgWindowCycles confidence-gated raw
         *  BPM estimates. m_avgBatchWriteIdx is the next slot to overwrite,
         *  m_avgBatchSize is how many valid entries it currently holds
         *  (<= kAvgWindowCycles, grows from 0 then stays full). */
        static float  m_avgBatch[kAvgWindowCycles];
        static size_t m_avgBatchSize;
        static size_t m_avgBatchWriteIdx;

        /** Parallel ring buffer of the confidence score that accompanied
         *  each accepted sample in m_avgBatch (same indices, same
         *  write/size counters) -- lets the display show the *average*
         *  confidence across the window instead of just the latest raw
         *  per-cycle value, which is jumpier than the averaged BPM it sits
         *  next to. */
        static float m_avgConfidenceBatch[kAvgWindowCycles];

        /** Current rolling-average HR. Recomputed every inference cycle
         *  from m_avgBatch -- not gated by the display throttle below. */
        static float m_smoothedBpm;

        /** Live mean confidence across the current averaging window (same
         *  indices/count as m_smoothedBpm's window) -- recomputed every
         *  cycle, latched into RhythmFormerResult::avgConfidence at the same
         *  throttle boundary as m_publishedHr so the two displayed numbers
         *  always describe the same window. */
        static float m_avgConfidence;

        /** Display-throttle state. m_msSinceLastDisplayUpdate accumulates
         *  via TickMs(); once it
         *  crosses kDisplayUpdateIntervalMs, m_publishedHr is refreshed from
         *  m_smoothedBpm and the counter resets. RhythmFormerResult::hrBpm mirrors
         *  m_publishedHr. */
        static uint32_t m_msSinceLastDisplayUpdate;
        static float    m_publishedHr;

        /** Design a 2nd-order Butterworth bandpass filter from
         *  [lowHz, highHz] at the given fps, in SOS form (2 second-order
         *  sections, each [b0,b1,b2,1,a1,a2]). Used by
         *  EstimateHrFromBuffer. */
        static void DesignButterBandpass(float fps, float lowHz, float highHz,
                                         ButterSos* sectionsOut, size_t& numSectionsOut);

        /** Zero-phase filter (filtfilt) over a float signal using the SOS
         *  cascade, with appropriate initial state reflection so the edges
         *  don't ring. Operates in-place on workBuf of length len. */
        static void FiltFiltSos(const ButterSos* sections, size_t numSections,
                                float* workBuf, size_t len);

        /** Single-section direct-form-II transposed filter (forward or
         *  reverse pass), writing output back into the same buffer. Used
         *  by FiltFiltSos. */
        static void FilterSosPass(const ButterSos& sec, float* buf, size_t len,
                                  bool reverse);

        /** Multiply signal in-place by a length-n Hann window. */
        static void ApplyHannWindow(float* buf, size_t len);

        static float EstimateHrFromBuffer(const float* signal,
                                          size_t len,
                                          float fps,
                                          float lowHz,
                                          float highHz,
                                          float& outConfidence);
    };

    /* -----------------------------------------------------------------------
     * Face-detection scheduling state-machine constants and struct.
     * -----------------------------------------------------------------------
     * The gate logic is:
     *   - After a positive detection, skip the next kFaceDetectSkipFrames
     *     frames before running the detector again (10-frame window: 1 run
     *     followed by 9 skipped).
     *   - If no face is detected for kFaceDetectNoFaceThreshold consecutive
     *     frames, trigger a recalibration that lasts kFaceDetectRecalibMs ms.
     *   - Independently, if motion is detected for kFaceDetectMotionThreshold
     *     consecutive frames, trigger the SAME recalibration hold -- subject
     *     movement corrupts the buffered rPPG diff-frames just as surely as
     *     the subject leaving frame does, even if a face is still present.
     *   - During recalibration, RhythmFormer inference is suppressed and the display
     *     shows "Face lost — Calibrating..."; the pipeline re-enters normal
     *     operation automatically once the hold expires.
     * ----------------------------------------------------------------------- */

    /** Frames to skip after a positive face detection (giving a 10-frame
     *  detection window: 1 run + 9 skipped).  Overridable at configure
     *  time via -Dalif_RhythmFormer_FACE_DETECT_SKIP_FRAMES=N. */
#if !defined(FACE_DETECT_SKIP_FRAMES) || ((FACE_DETECT_SKIP_FRAMES + 0) < 0)
#undef FACE_DETECT_SKIP_FRAMES
#define FACE_DETECT_SKIP_FRAMES 9
#endif
    static constexpr size_t kFaceDetectSkipFrames = FACE_DETECT_SKIP_FRAMES;

    /** Consecutive frames without a face before recalibration is triggered. */
#if !defined(FACE_DETECT_NO_FACE_THRESHOLD) || ((FACE_DETECT_NO_FACE_THRESHOLD + 0) <= 0)
#undef FACE_DETECT_NO_FACE_THRESHOLD
#define FACE_DETECT_NO_FACE_THRESHOLD 20
#endif
    static constexpr size_t kFaceDetectNoFaceThreshold = FACE_DETECT_NO_FACE_THRESHOLD;

    /** Consecutive frames of detected motion before recalibration triggers.
     *  Default 1: a single motion-positive frame immediately restarts
     *  calibration, since even brief subject movement corrupts the rPPG
     *  diff-frame data already buffered. Raise via
     *  -Dalif_RhythmFormer_FACE_DETECT_MOTION_THRESHOLD=N if a single frame proves
     *  too sensitive in practice (e.g. blinks or micro-movements
     *  false-triggering recalibration). */
#if !defined(FACE_DETECT_MOTION_THRESHOLD) || ((FACE_DETECT_MOTION_THRESHOLD + 0) <= 0)
#undef FACE_DETECT_MOTION_THRESHOLD
#define FACE_DETECT_MOTION_THRESHOLD 1
#endif
    static constexpr size_t kFaceDetectMotionThreshold = FACE_DETECT_MOTION_THRESHOLD;

    /** Recalibration hold duration in milliseconds (5 seconds). */
    static constexpr uint32_t kFaceDetectRecalibMs = 5000U;

    /**
     * @brief   Lightweight state machine that encapsulates the face-detection
     *          scheduling policy for the RhythmFormer pipeline.
     *
     *          Callers should:
     *          1. Call ShouldRunDetection() at every camera frame.
     *          2. If true, run FomoFaceDetector::DetectFace() and call
     *             OnFaceDetected() or OnNoFaceDetected() with the result;
     *             also run MotionDetector::DetectMotion() and call
     *             OnMotionDetected() or OnNoMotionDetected() with that
     *             result. The two are independent -- either one alone can
     *             trigger recalibration.
     *          3. If OnNoFaceDetected() or OnMotionDetected() returns true,
     *             call BeginRecalibration() and reset the RhythmFormer pipeline.
     *          4. Call TickMs(elapsedMs) every frame to advance the recalib
     *             hold timer; check recalibrating to gate RhythmFormer inference.
     *          5. TickMs() returns true exactly once when the hold expires;
     *             the caller should then resume normal inference.
     */
    struct FaceDetectionState {
        /** True while the recalibration hold (kFaceDetectRecalibMs) is
         *  active. */
        bool     recalibrating        = false;

        /** Countdown of frames still to be skipped before the next
         *  detection run. 0 means: run detection on this frame. */
        size_t   skipFramesRemaining  = 0;

        /** Number of consecutive camera frames with no face detected.
         *  Reset to 0 on every positive detection or recalib start. */
        size_t   noFaceStreak         = 0;

        /** Number of consecutive camera frames with motion detected.
         *  Reset to 0 the moment a frame with no motion is observed, or
         *  when recalibration begins. */
        size_t   motionStreak         = 0;

        /** Remaining milliseconds in the current recalibration hold.
         *  Decremented by TickMs(); set to kFaceDetectRecalibMs by
         *  BeginRecalibration(). */
        uint32_t recalibMsRemaining   = 0U;

        /** Returns true when face/motion detection should be run on this
         *  frame. False during the skip window after a positive detection,
         *  and false while recalibrating (no inference is needed). */
        bool ShouldRunDetection() const;

        /** Record a positive detection.  Resets the no-face streak and
         *  arms the skip-frame counter so the next kFaceDetectSkipFrames
         *  frames are not checked. */
        void OnFaceDetected();

        /** Record a negative detection result for the current frame.
         *  Returns true the moment the no-face streak reaches
         *  kFaceDetectNoFaceThreshold — the caller should then call
         *  BeginRecalibration() and reset the RhythmFormer pipeline. */
        bool OnNoFaceDetected();

        /** Record a motion-positive result for the current frame. Returns
         *  true the moment the motion streak reaches
         *  kFaceDetectMotionThreshold — the caller should then call
         *  BeginRecalibration() and reset the RhythmFormer pipeline, exactly as
         *  with OnNoFaceDetected(). Independent of the face streak: motion
         *  can trigger recalibration even while a face is present. */
        bool OnMotionDetected();

        /** Record a motion-negative result for the current frame. Resets
         *  the motion streak back to 0. */
        void OnNoMotionDetected();

        /** Start the kFaceDetectRecalibMs recalibration hold.  Called by
         *  the use-case handler when OnNoFaceDetected() or
         *  OnMotionDetected() returns true. */
        void BeginRecalibration();

        /** Advance the wall-clock timer driving the recalibration hold.
         *  Should be called once per camera frame with the elapsed time
         *  since the previous call.
         *  Returns true exactly once: when the hold period expires.  The
         *  caller must then resume RhythmFormer inference; the state machine
         *  automatically exits recalibrating mode at that point. */
        bool TickMs(uint32_t elapsedMs);

        /** Full reset: clears all counters and exits recalibration mode.
         *  Equivalent to re-constructing the struct with default values. */
        void Reset();
    };

} /* namespace app */
} /* namespace arm */

#endif /* RHYTHMFORMER_PROCESSING_HPP */