#include "mlek/use_case/rhythmformer/MotionDetector.hpp"

#include <cstring>

namespace arm {
namespace app {

    /* ---- Static storage definitions for the class ---- */
    uint8_t MOTIONDET_DRAM_ATTR MotionDetector::s_curLuma[MOTION_DETECT_H][MOTION_DETECT_W];
    uint8_t MOTIONDET_DRAM_ATTR MotionDetector::s_prevLuma[MOTION_DETECT_H][MOTION_DETECT_W];
    bool    MotionDetector::s_havePrevFrame = false;

    /* ------------------------------------------------------------------
     * Motion gate: luma abs-diff over a downsampled grid.
     * Identical algorithm to HybridFaceDetector::MotionGate — see that
     * file's history for rationale — just lifted out into its own TU with
     * its own static storage so it can be used without the rest of the
     * hybrid pipeline (skin-tone gate, Haar cascade).
     * ------------------------------------------------------------------ */
    bool MotionDetector::DetectMotion(const uint8_t* rgb888, size_t srcW, size_t srcH)
    {
        if (rgb888 == nullptr || srcW == 0U || srcH == 0U) {
            return false;
        }

        /* Step 1: nearest-neighbour downsample RGB888 → kMotionW×kMotionH
         * luma. BT.601 weights in Q8 fixed point. */
        for (int dy = 0; dy < kMotionH; ++dy) {
            const size_t sy =
                (static_cast<size_t>(dy) * srcH) / static_cast<size_t>(kMotionH);
            const uint8_t* srcRow = rgb888 + sy * srcW * 3U;
            uint8_t* dstRow = &s_curLuma[dy][0];
            for (int dx = 0; dx < kMotionW; ++dx) {
                const size_t sx =
                    (static_cast<size_t>(dx) * srcW) / static_cast<size_t>(kMotionW);
                const uint8_t* px = srcRow + sx * 3U;
                const uint32_t gray =
                    (static_cast<uint32_t>(px[0]) * 77u +
                     static_cast<uint32_t>(px[1]) * 151u +
                     static_cast<uint32_t>(px[2]) * 28u) >> 8;
                dstRow[dx] = static_cast<uint8_t>(gray);
            }
        }

        /* Step 2: if we have a previous frame, count pixels whose luma
         * changed by >= kMotionDiff. On the very first call (no previous
         * frame yet) we can't detect motion at all; kBypassMotionFirst
         * decides what that reports as. */
        int moved = 0;
        const int total = kMotionW * kMotionH;
        const bool hadPrevFrame = s_havePrevFrame;
        if (hadPrevFrame) {
            const uint8_t* c = &s_curLuma[0][0];
            const uint8_t* p = &s_prevLuma[0][0];
            for (int i = 0; i < total; ++i) {
                const int d = static_cast<int>(c[i]) - static_cast<int>(p[i]);
                const int ad = d < 0 ? -d : d;
                if (ad >= kMotionDiff) {
                    ++moved;
                }
            }
        }

        /* Step 3: roll history so the NEXT call has a baseline. */
        std::memcpy(&s_prevLuma[0][0], &s_curLuma[0][0],
                    static_cast<size_t>(total) * sizeof(uint8_t));
        s_havePrevFrame = true;

        /* Step 4: decision. */
        if (!hadPrevFrame) {
            return kBypassMotionFirst ? false : true;
        }
        if (kMotionMinPct <= 0) {
            return false; /* motion gate disabled -> never reports motion */
        }
        const int threshold = (total * kMotionMinPct) / 100;
        return moved >= threshold;
    }

    void MotionDetector::ResetMotionHistory()
    {
        s_havePrevFrame = false;
    }

} /* namespace app */
} /* namespace arm */
