#ifndef MOTION_DETECTOR_HPP
#define MOTION_DETECTOR_HPP

#include <cstddef>
#include <cstdint>

/* ---------------------------------------------------------------------------
 * Standalone extraction of HybridFaceDetector's Stage-1 motion gate.
 *
 * RhythmFormer runs FomoFaceDetector (NPU) for "is a face present" and this class
 * for "did the subject move" — the two are complementary, not competing:
 * FOMO answers presence, MotionDetector answers stillness, and rPPG needs
 * BOTH (a present, STILL subject) for the buffered diff-frames to represent
 * real cardiac signal instead of motion artifact. HybridFaceDetector's
 * skin-tone gate and Haar cascade are intentionally NOT pulled in here.
 * ------------------------------------------------------------------------- */

#if defined(__GNUC__) || defined(__clang__)
#define MOTIONDET_DRAM_ATTR \
    __attribute__((section(".bss.NoInit.motiondet_buf_dram"), aligned(16)))
#else
#define MOTIONDET_DRAM_ATTR
#endif

/* ---------- Compile-time tuning knobs (overridable via CMake) ----------
 * Defaults intentionally mirror HYBRID_FD_SKIN_W/H, HYBRID_FD_MOTION_MIN_PCT,
 * HYBRID_FD_MOTION_DIFF and HYBRID_FD_BYPASS_MOTION_ON_FIRST so behaviour is
 * unchanged from what HybridFaceDetector's Stage 1 used to do — only the
 * name changed, to make clear this is now an independent, standalone gate
 * rather than a private stage inside the hybrid pipeline. */
#if defined(MOTION_DETECT_W) && ((MOTION_DETECT_W + 0) <= 0)
#undef MOTION_DETECT_W
#endif
#ifndef MOTION_DETECT_W
#define MOTION_DETECT_W 64
#endif

#if defined(MOTION_DETECT_H) && ((MOTION_DETECT_H + 0) <= 0)
#undef MOTION_DETECT_H
#endif
#ifndef MOTION_DETECT_H
#define MOTION_DETECT_H 64
#endif

/* Minimum fraction of the downsampled grid that must differ by >=
 * MOTION_DETECT_DIFF luma counts from the previous frame before the frame
 * counts as "motion detected". 1% = ~41 px out of a 64x64 grid — enough to
 * reject sensor read noise while still catching a head twitch or blink. */
#ifndef MOTION_DETECT_MIN_PCT
#define MOTION_DETECT_MIN_PCT 1
#endif

/* Luma delta (0-255) for a pixel to count as "moved". */
#if defined(MOTION_DETECT_DIFF) && ((MOTION_DETECT_DIFF + 0) <= 0)
#undef MOTION_DETECT_DIFF
#endif
#ifndef MOTION_DETECT_DIFF
#define MOTION_DETECT_DIFF 15
#endif

/* If true, the very first DetectMotion() call of a session (no previous
 * frame to diff against) is reported as "no motion" instead of forcing a
 * false-positive motion trigger on boot / right after a recalibration
 * resets history. Leave at 1 unless the caller guarantees DetectMotion is
 * never invoked on a freshly-reset detector. */
#ifndef MOTION_DETECT_BYPASS_ON_FIRST
#define MOTION_DETECT_BYPASS_ON_FIRST 1
#endif

namespace arm {
namespace app {

/**
 * @brief   Standalone luma-diff motion detector (HybridFaceDetector's old
 *          Stage 1, extracted so it can run alongside FomoFaceDetector
 *          without pulling in the skin-tone gate or Haar cascade).
 *
 *          Downsamples the incoming RGB888 frame to a small luma grid,
 *          diffs it against the previous call's grid, and reports motion
 *          if the fraction of pixels that changed by more than
 *          kMotionDiff luma counts meets or exceeds kMotionMinPct.
 *
 *          All methods are static; the class has no per-instance state.
 */
class MotionDetector {
public:
    /** Run one motion-detection pass.
     *
     * @param[in] rgb888   Canonical RGB888 source frame (R first, 3 Bpp).
     * @param[in] width    Frame width in pixels.
     * @param[in] height   Frame height in pixels.
     *
     * @return  true if the fraction of changed pixels vs. the previous
     *          call's frame meets or exceeds kMotionMinPct. Always false
     *          on the first call after construction / ResetMotionHistory()
     *          when kBypassMotionFirst is true (the default). */
    static bool DetectMotion(const uint8_t* rgb888, size_t width, size_t height);

    /** Reset motion-history state (call on init, and after a
     *  recalibration begins, so the next DetectMotion() call doesn't diff
     *  against a stale pre-recalibration frame). */
    static void ResetMotionHistory();

private:
    static constexpr int kMotionW = MOTION_DETECT_W;
    static constexpr int kMotionH = MOTION_DETECT_H;

    static constexpr int  kMotionMinPct      = MOTION_DETECT_MIN_PCT;
    static constexpr int  kMotionDiff        = MOTION_DETECT_DIFF;
    static constexpr bool kBypassMotionFirst = (MOTION_DETECT_BYPASS_ON_FIRST != 0);

    /* 8-bit luma for the current + previous kMotionW×kMotionH grid. */
    static uint8_t MOTIONDET_DRAM_ATTR s_curLuma[MOTION_DETECT_H][MOTION_DETECT_W];
    static uint8_t MOTIONDET_DRAM_ATTR s_prevLuma[MOTION_DETECT_H][MOTION_DETECT_W];

    static bool s_havePrevFrame;
};

} /* namespace app */
} /* namespace arm */

#endif /* MOTION_DETECTOR_HPP */
