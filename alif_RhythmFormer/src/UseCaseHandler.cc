#include "UseCaseHandler.hpp"
#include "UseCaseCommonUtils.hpp"
#include "mlek/fwk/tflm/RhythmFormerModel.hpp"
#include "mlek/use_case/rhythmformer/RhythmFormerProcessing.hpp"
#include "mlek/use_case/rhythmformer/FomoFaceDetector.hpp"
#include "mlek/use_case/rhythmformer/MotionDetector.hpp"
#include "hal.h"
#include "mlek/log/log_macros.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "lvgl.h"
#include "lv_port.h"
#include "lv_paint_utils.h"
#include "ScreenLayout.hpp"

#ifndef CAMERA_WIDTH
#define CAMERA_WIDTH 240
#endif

#ifndef CAMERA_HEIGHT
#define CAMERA_HEIGHT 240
#endif

#ifndef RHYTHMFORMER_ASSUMED_FPS
#define RHYTHMFORMER_ASSUMED_FPS 30.0f
#endif

/* The Alif camera HAL outputs BGR order on the MIPI-CSI bus for all sensors.
 * Swap R and B bytes during snapshot normalization so downstream code (LVGL
 * display + the RhythmFormer 72x72 CNN) always receives canonical RGB order. If you
 * ever wire a sensor with RGB order on the data lanes, change this to 0. */
#ifndef RHYTHMFORMER_SWAP_RB_CHANNELS
#define RHYTHMFORMER_SWAP_RB_CHANNELS 1
#endif

/* HAL rgb buffer caps per sensor (from image_processing.h):
 *   MT9M114 = 320x320 MAX, OV5675 = 480x480 MAX, ARX3A0 = 560x560 MAX.
 * 240x240 fits ALL sensors comfortably and is already square, so the 72x72
 * center-crop resize in RhythmFormerPreProcess becomes a clean 3.333x bilinear
 * downscale with zero pixel waste. */
#if (CAMERA_WIDTH > 240) || (CAMERA_HEIGHT > 240)
#error "alif_RhythmFormer: CAMERA_WIDTH/HEIGHT must not exceed 240 on any sensor (MT9M114 caps at 320)."
#endif

#define MIMAGE_X CAMERA_WIDTH
#define MIMAGE_Y CAMERA_HEIGHT
#define MIMAGE_RGB_BYTES (MIMAGE_X * MIMAGE_Y * 3U)

/* Native 1:1 LVGL canvas — no zoom = each HAL buffer pixel maps 1:1 to screen
 * pixels, so no nearest-neighbor block artifacts. Bilinear antialias in
 * ScreenLayout.cc keeps the flex-box container downscale artifact-free if
 * the layout squeezes the image slightly smaller than 240px. */
#define LIMAGE_X MIMAGE_X
#define LIMAGE_Y MIMAGE_Y
#define LV_ZOOM  (1 * 256)

namespace {

lvgl_pixel_t lvgl_image[LIMAGE_Y][LIMAGE_X] __attribute__((section(".bss.lcd_image_buf")));

/* g_cameraSnapshot is ALWAYS normalized to RGB888 (3 bytes/pixel, R-first
 * canonical order) by SnapshotCameraFrame() regardless of what the HAL
 * actually delivered. Downstream LVGL display copy + RhythmFormerPreProcess CNN
 * input can rely on this invariant. */
uint8_t g_cameraSnapshot[MIMAGE_RGB_BYTES] __attribute__((section(".bss.camera_frame_buf")));


uint32_t g_lastHandlerCycles    = 0;
bool     g_loggedFrameFormatOnce = false;

/* ---------------------------------------------------------------------------
 * Single top-level pipeline state, computed once per handler cycle and
 * emitted as a structured [RHYTHMFORMER_STATE] log line so an external UI
 * can drive a status indicator without parsing the more granular
 * [RHYTHMFORMER_FACE]/[RHYTHMFORMER_MOTION]/[RHYTHMFORMER_UI] lines.
 *
 * Precedence (first match wins): camera error > recalibrating > this-cycle
 * motion/no-face > low light > buffering > tracking.
 * ------------------------------------------------------------------------- */
enum class RhythmFormerState : uint8_t {
    kCamError = 0,
    kCalibrating,
    kMotion,
    kNoFace,
    kLowLight,
    kBuffering,
    kTracking,
};

const char* ToString(RhythmFormerState state)
{
    switch (state) {
        case RhythmFormerState::kCamError:    return "CAM_ERROR";
        case RhythmFormerState::kCalibrating: return "CALIBRATING";
        case RhythmFormerState::kMotion:      return "MOTION";
        case RhythmFormerState::kNoFace:      return "NO_FACE";
        case RhythmFormerState::kLowLight:    return "LOW_LIGHT";
        case RhythmFormerState::kBuffering:   return "BUFFERING";
        case RhythmFormerState::kTracking:    return "TRACKING";
    }
    return "UNKNOWN";
}

/** Emit one structured, machine-parseable state line per handler cycle.
 *  Flat key=value pairs so a UI can split on whitespace without a real
 *  parser; `state` alone drives a simple status badge. */
void LogRhythmFormerState(RhythmFormerState state,
                   const arm::app::RhythmFormerResult& result,
                   const arm::app::FaceDetectionState& faceState,
                   bool lowLight)
{
    info("[RHYTHMFORMER_STATE] frame=%zu chunk=%zu state=%s "
         "recalibrating=%d recalib_ms_left=%lu "
         "no_face_streak=%zu motion_streak=%zu low_light=%d "
         "hr_valid=%d hr_bpm=%.2f hr_conf=%.4f\n",
         result.totalFrames,
         result.chunkIndex,
         ToString(state),
         faceState.recalibrating ? 1 : 0,
         static_cast<unsigned long>(faceState.recalibMsRemaining),
         faceState.noFaceStreak,
         faceState.motionStreak,
         lowLight ? 1 : 0,
         result.hrValid ? 1 : 0,
         static_cast<double>(result.hrBpm),
         static_cast<double>(result.hrConfidence));
}

/** Pack canonical RGB888 source pixels (R, G, B bytes) into the LVGL
 *  depth-specific framebuffer. Uses lv_color_make() + a 1-pixel memcpy since
 *  the generic lv_color_t union is a different C++ type than lv_color16_t /
 *  lv_color32_t on LV_COLOR_DEPTH-specific builds — the union's leading N
 *  bytes hold exactly the packed bits we need, and memcpy avoids the C++
 *  operator= type mismatch. */
void CopyRgb888ToLvgl(int width, int height, const uint8_t* src, lvgl_pixel_t* dst)
{
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t si         = (static_cast<size_t>(y) * static_cast<size_t>(width) +
                               static_cast<size_t>(x)) * 3U;
            const lv_color_t packed = lv_color_make(src[si], src[si + 1], src[si + 2]);
            std::memcpy(&dst[y * width + x], &packed, sizeof(lvgl_pixel_t));
        }
    }
}

/** Takes the raw HAL buffer + reported size, auto-detects the on-wire pixel
 *  format from (capturedFrameSize / (width*height)), normalizes everything
 *  to canonical RGB888 order in g_cameraSnapshot, and logs a one-time banner
 *  of the detected format on UART.
 *
 *  Returns true if a valid format was detected and g_cameraSnapshot is
 *  refreshed. Returns false on size/format mismatch without touching
 *  g_cameraSnapshot (caller retains the last valid frame). */
bool SnapshotCameraFrame(const uint8_t* imageData, uint32_t capturedFrameSize)
{
    const uint32_t pixelCount = static_cast<uint32_t>(MIMAGE_X) *
                                static_cast<uint32_t>(MIMAGE_Y);
    if (pixelCount == 0U) {
        printf_err("SnapshotCameraFrame: zero pixel count? %dx%d invalid\n",
                   MIMAGE_X, MIMAGE_Y);
        return false;
    }
    const uint32_t bpp = capturedFrameSize / pixelCount;

    /* ---- Accept: bpp=2 (RGB565, MT9M114 sensor, CIMAGE_USE_RGB565=1 in HAL)
     *          OR: bpp=3 (RGB/BGR 888, OV5675/ARX3A0 sensors)             ---- */
    if (bpp != 2U && bpp != 3U) {
        printf_err(
            "Camera frame format unrecognised: %lu bytes = %lu bpp for %dx%d px.\n"
            "Expected bpp=2 (RGB565 / MT9M114) or bpp=3 (RGB/BGR888 / OV5675).\n"
            "This usually means stale CMake USER_OPTION cache — delete CMakeCache.txt and rebuild:\n"
            "  rm -rf build-alif-e8\n"
            "  cmake ... -Dalif_RhythmFormer_CAMERA_WIDTH=%d -Dalif_RhythmFormer_CAMERA_HEIGHT=%d\n",
            static_cast<unsigned long>(capturedFrameSize),
            static_cast<unsigned long>(bpp),
            MIMAGE_X, MIMAGE_Y,
            MIMAGE_X, MIMAGE_Y);
        return false;
    }

    if (!g_loggedFrameFormatOnce) {
        info("RhythmFormer camera: %lu bytes @ %dx%d → detected bpp=%lu (%s, RB-swap=%s). Normalising to RGB888 [%lu bytes snapshot]\n",
             static_cast<unsigned long>(capturedFrameSize),
             MIMAGE_X, MIMAGE_Y,
             static_cast<unsigned long>(bpp),
             (bpp == 2U) ? "RGB565 MT9M114" : "RGB/BGR888 OV5675",
             (RHYTHMFORMER_SWAP_RB_CHANNELS != 0) ? "ON" : "OFF",
             static_cast<unsigned long>(MIMAGE_RGB_BYTES));
        g_loggedFrameFormatOnce = true;
    }

    /* ---------- Normalise: unpack + BGR→RGB swap into g_cameraSnapshot. ---------- */
    if (bpp == 2U) {
        /* ---- RGB565: two bytes per pixel, little-endian 16-bit word ----
         *       bits:  R[4:0] G[5:0] B[4:0]
         *   Alif MIPI lane order flips R and B, so swap those in the unpack. */
        for (uint32_t i = 0U; i < pixelCount; ++i) {
            const uint32_t srcI = i * 2U;
            const uint16_t w565 = static_cast<uint16_t>(imageData[srcI]) |
                                  (static_cast<uint16_t>(imageData[srcI + 1]) << 8);
            uint8_t r5 = static_cast<uint8_t>((w565 >> 11U) & 0x1FU);
            uint8_t g6 = static_cast<uint8_t>((w565 >>  5U) & 0x3FU);
            uint8_t b5 = static_cast<uint8_t>((w565 >>  0U) & 0x1FU);
            r5 = static_cast<uint8_t>((r5 << 3U) | (r5 >> 2U));   // 5→8 bit expand
            g6 = static_cast<uint8_t>((g6 << 2U) | (g6 >> 4U));   // 6→8 bit expand
            b5 = static_cast<uint8_t>((b5 << 3U) | (b5 >> 2U));   // 5→8 bit expand
#if (RHYTHMFORMER_SWAP_RB_CHANNELS != 0)
            std::swap(r5, b5);
#endif
            const uint32_t dstI = i * 3U;
            g_cameraSnapshot[dstI + 0U] = r5;
            g_cameraSnapshot[dstI + 1U] = g6;
            g_cameraSnapshot[dstI + 2U] = b5;
        }
    } else {
        /* ---- RGB/BGR888: three bytes per pixel. ----
         *   Just copy verbatim and swap R↔B if configured. */
        for (uint32_t i = 0U; i < pixelCount; ++i) {
            const uint32_t srcI = i * 3U;
            uint8_t c0 = imageData[srcI + 0U];
            const uint8_t c1 = imageData[srcI + 1U];
            uint8_t c2 = imageData[srcI + 2U];
#if (RHYTHMFORMER_SWAP_RB_CHANNELS != 0)
            std::swap(c0, c2);
#endif
            const uint32_t dstI = srcI;
            g_cameraSnapshot[dstI + 0U] = c0;
            g_cameraSnapshot[dstI + 1U] = c1;
            g_cameraSnapshot[dstI + 2U] = c2;
        }
    }
    return true;
}

} /* anonymous namespace */

namespace alif {
namespace app {

    using namespace arm::app;

    static void UpdateHrDisplay(const RhythmFormerResult& result)
    {
        /* Show the throttled, confidence-averaged value (result.hrBpm), not
         * the jittery per-cycle raw estimate. */
        if (!result.hrValid) {
            lv_label_set_text_fmt(ScreenLayoutLabelObject(1), "HR: Calibrating...");
        } else if (result.hrBpm > 0.0f) {
            lv_label_set_text_fmt(ScreenLayoutLabelObject(1), "HR: %.1f BPM (conf %.0f%%)",
                                  static_cast<double>(result.hrBpm),
                                  static_cast<double>(result.avgConfidence) * 100.0);
        } else {
            /* Calibrated, but no confidence-gated reading yet. Fall back to
             * the raw per-cycle confidence since the averaging window is
             * still empty (avgConfidence would just read 0). */
            lv_label_set_text_fmt(ScreenLayoutLabelObject(1), "HR: -- BPM (conf %.0f%%)",
                                  static_cast<double>(result.hrConfidence) * 100.0);
        }
    }

    bool RhythmFormerHandlerInit(arm::app::fwk::tflm::RhythmFormerModel& /*model*/)
    {
        ScreenLayoutInit(lvgl_image, sizeof(lvgl_image), LIMAGE_X, LIMAGE_Y, LV_ZOOM);

        uint32_t lv_lock_state = lv_port_lock();
        lv_label_set_text_static(ScreenLayoutHeaderObject(), "RhythmFormer rPPG");
        lv_label_set_text_static(ScreenLayoutLabelObject(0), "Buffering frames...");
        lv_label_set_text_static(ScreenLayoutLabelObject(1), "HR: -- BPM");
        lv_port_unlock(lv_lock_state);

        info("RhythmFormer rPPG: camera %dx%d, LVGL canvas %dx%d, zoom=%u\n",
             MIMAGE_X,
             MIMAGE_Y,
             LIMAGE_X,
             LIMAGE_Y,
             static_cast<unsigned>(LV_ZOOM));
        info("RhythmFormer rPPG: HAL rgb buffer caps — MT9M114=320, OV5675=480 (keep capture <=240)\n");

        if (!hal_camera_init()) {
            printf_err("hal_camera_init failed!\n");
            return false;
        }

        if (!hal_camera_configure(MIMAGE_X,
                                  MIMAGE_Y,
                                  HAL_CAMERA_MODE_SINGLE_FRAME,
                                  HAL_CAMERA_COLOUR_FORMAT_RGB888)) {
            printf_err("hal_camera_configure(%dx%d) failed.\n", MIMAGE_X, MIMAGE_Y);
            return false;
        }

        MotionDetector::ResetMotionHistory();

        info("RhythmFormer rPPG: camera ready.\n");
        return true;
    }

    bool RhythmFormerHandler(ApplicationContext& ctx)
    {
        /* ---------------------------------------------------------------
         * Measure time elapsed since the last handler call.  Used by both
         * RhythmFormerPostProcess::TickMs (HR display throttle) and
         * FaceDetectionState::TickMs (recalibration hold countdown).
         * --------------------------------------------------------------- */
        const uint32_t nowCycles = Get_SysTick_Cycle_Count32();
        uint32_t deltaMs = 0U;
        if (g_lastHandlerCycles != 0) {
            const uint32_t delta = nowCycles - g_lastHandlerCycles;
            deltaMs = static_cast<uint32_t>(
                (static_cast<uint64_t>(delta) * 1000ULL) /
                static_cast<uint64_t>(SystemCoreClock > 0 ? SystemCoreClock : 1));
            ctx.Get<RhythmFormerPostProcess&>("postProcess").TickMs(deltaMs);
        }
        g_lastHandlerCycles = nowCycles;

        auto& profiler    = ctx.Get<Profiler&>("profiler");
        auto& model       = ctx.Get<fwk::iface::Model&>("model");
        auto& preProcess  = ctx.Get<RhythmFormerPreProcess&>("preProcess");
        auto& postProcess = ctx.Get<RhythmFormerPostProcess&>("postProcess");
        auto& result      = ctx.Get<RhythmFormerResult&>("result");
        auto& faceState   = ctx.Get<FaceDetectionState&>("faceState");

        if (!model.IsInited()) {
            printf_err("Model is not initialised!\n");
            return false;
        }

        /* ---------------------------------------------------------------
         * Advance the face-detection state machine wall-clock timer.
         * If TickMs returns true the 30-second recalibration hold has
         * just expired — log it but remain in normal flow so the camera
         * frame captured below becomes the first frame of the new session.
         * --------------------------------------------------------------- */
        if (faceState.TickMs(deltaMs)) {
            info("[RHYTHMFORMER_FACE] Recalibration hold expired — resuming RhythmFormer inference.\n");
        }

        /* ---------------------------------------------------------------
         * Camera capture.
         * --------------------------------------------------------------- */
        if (!hal_camera_start()) {
            printf_err("hal_camera_start failed\n");
            return false;
        }

        uint32_t capturedFrameSize = 0;
        const uint8_t* imageData   = hal_camera_get_captured_frame(&capturedFrameSize);
        if (!imageData || !capturedFrameSize) {
            printf_err("hal_camera_get_captured_frame failed\n");
            return false;
        }

        bool snapshotOk = SnapshotCameraFrame(imageData, capturedFrameSize);

        /* Always refresh the LVGL camera preview first — even on a snapshot
         * failure we leave the LAST KNOWN GOOD frame visible so the user
         * never stares at BSS-initialised pixel garbage.  Only skip the heavy
         * inference / preprocess pipeline if the frame was invalid. */
        {
            ScopedLVGLLock lv_lock;
            CopyRgb888ToLvgl(MIMAGE_X, MIMAGE_Y, g_cameraSnapshot, &lvgl_image[0][0]);
            lv_obj_invalidate(ScreenLayoutImageObject());

            if (!run_requested()) {
                lv_led_off(ScreenLayoutLEDObject());
                return true;
            }
            lv_led_on(ScreenLayoutLEDObject());
        }

        if (!snapshotOk) {
            /* Format/size mismatch — display the frame counter/last HR so
             * the UI still updates, but skip inference until the HAL starts
             * delivering the expected formats again. */
            LogRhythmFormerState(RhythmFormerState::kCamError, result, faceState, false);
            ScopedLVGLLock lv_lock;
            lv_label_set_text_fmt(ScreenLayoutLabelObject(0), "Frames: %zu [cam fmt ERR]",
                                  result.totalFrames);
            UpdateHrDisplay(result);
            return true;
        }

        ++result.totalFrames;

        /* Tracks the single "current state" this cycle for the structured
         * [RHYTHMFORMER_STATE] log at the bottom of the handler. Overwritten as we
         * learn more; only ever moves to a HIGHER-precedence state (see the
         * RhythmFormerState/LogRhythmFormerState comment above) than whatever it already
         * holds, except where noted. */
        RhythmFormerState frameState = RhythmFormerState::kTracking;

        /* ---------------------------------------------------------------
         * Face + motion detection — FOMO 72x72 int8 model (NPU-accelerated)
         * for presence, MotionDetector (CPU luma-diff) for stillness.
         *
         * Policy (FaceDetectionState):
         *   • ShouldRunDetection() is false during the post-detection skip
         *     window or while recalibrating — both detectors are skipped.
         *   • A positive face detection arms the skip counter.
         *   • kFaceDetectNoFaceThreshold consecutive no-face frames, OR
         *     kFaceDetectMotionThreshold consecutive motion frames (default
         *     1, since movement corrupts buffered rPPG diff-frames even
         *     with a face present), each independently trigger the same
         *     recalibration hold and reset the RhythmFormer pipeline.
         * --------------------------------------------------------------- */
        if (faceState.ShouldRunDetection()) {
            auto& faceModel = ctx.Get<fwk::iface::Model&>("faceModel");

            const uint32_t ufStart = Get_SysTick_Cycle_Count32();
            const bool faceDetected = FomoFaceDetector::DetectFace(
                faceModel, g_cameraSnapshot, MIMAGE_X, MIMAGE_Y);
            const uint32_t ufCycles = Get_SysTick_Cycle_Count32() - ufStart;
            const double ufMs =
                (static_cast<double>(ufCycles) /
                 static_cast<double>(SystemCoreClock > 0 ? SystemCoreClock : 1)) *
                1000.0;

#if ENABLE_MOTION_DETECTOR
            const uint32_t mdStart = Get_SysTick_Cycle_Count32();
            const bool motionDetected = MotionDetector::DetectMotion(
                g_cameraSnapshot, MIMAGE_X, MIMAGE_Y);
            const uint32_t mdCycles = Get_SysTick_Cycle_Count32() - mdStart;
            const double mdMs =
                (static_cast<double>(mdCycles) /
                 static_cast<double>(SystemCoreClock > 0 ? SystemCoreClock : 1)) *
                1000.0;
#else
            /* Motion detector disabled at compile time (ENABLE_MOTION_DETECTOR=OFF):
             * skip the downsample+diff work entirely and always report "no
             * motion" — recalibration can then only be triggered by face loss. */
            const bool motionDetected = false;
            const double mdMs = 0.0;
#endif

            bool triggerRecalib = false;
            const char* recalibCause = "";

            if (faceDetected) {
                faceState.OnFaceDetected();
                info("[RHYTHMFORMER_FACE] detected=1 inf_ms=%.2f skip=%zu no_face_streak=0 recalib=0 [fomo]\n",
                     ufMs,
                     faceState.skipFramesRemaining);
            } else {
                frameState = RhythmFormerState::kNoFace;
                const bool faceTrigger = faceState.OnNoFaceDetected();
                info("[RHYTHMFORMER_FACE] detected=0 inf_ms=%.2f skip=0 no_face_streak=%zu recalib=%d [fomo]\n",
                     ufMs,
                     faceState.noFaceStreak, faceTrigger ? 1 : 0);
                if (faceTrigger) {
                    triggerRecalib = true;
                    recalibCause = "no_face";
                }
            }

            if (motionDetected) {
                frameState = RhythmFormerState::kMotion;
                const bool motionTrigger = faceState.OnMotionDetected();
                info("[RHYTHMFORMER_MOTION] detected=1 inf_ms=%.2f motion_streak=%zu recalib=%d\n",
                     mdMs,
                     faceState.motionStreak, motionTrigger ? 1 : 0);
                if (motionTrigger) {
                    triggerRecalib = true;
                    recalibCause = (recalibCause[0] != '\0') ? "no_face+motion" : "motion";
                }
            } else {
                faceState.OnNoMotionDetected();
                info("[RHYTHMFORMER_MOTION] detected=0 inf_ms=%.2f motion_streak=0\n", mdMs);
            }

            if (triggerRecalib) {
                info("[RHYTHMFORMER_FACE] Recalibration triggered (cause=%s) — starting "
                     "%u ms recalibration hold.\n",
                     recalibCause,
                     static_cast<unsigned>(kFaceDetectRecalibMs));

                faceState.BeginRecalibration();
                postProcess.ResetAll();
                preProcess.Reset();
                MotionDetector::ResetMotionHistory();

                LogRhythmFormerState(RhythmFormerState::kCalibrating, result, faceState, false);

                /* Show recalibration status and return early; RhythmFormer
                 * inference is suppressed for the duration of the hold. */
                ScopedLVGLLock lv_lock;
                lv_label_set_text_fmt(ScreenLayoutLabelObject(0),
                                     "Frames: %zu  %s", result.totalFrames,
                                     (std::strcmp(recalibCause, "motion") == 0)
                                         ? "Motion detected"
                                         : "Face lost");
                lv_label_set_text_static(ScreenLayoutLabelObject(1),
                                         "HR: Calibrating...");
                return true;
            }
        }

        /* ---------------------------------------------------------------
         * While in a recalibration hold, suppress RhythmFormer inference and
         * update the display with the calibration status + countdown.
         * --------------------------------------------------------------- */
        if (faceState.recalibrating) {
            const uint32_t secsLeft =
                (faceState.recalibMsRemaining + 999U) / 1000U;
            LogRhythmFormerState(RhythmFormerState::kCalibrating, result, faceState, false);
            ScopedLVGLLock lv_lock;
            lv_label_set_text_fmt(ScreenLayoutLabelObject(0),
                                  "Frames: %zu  Calibrating (%lu s)",
                                  result.totalFrames,
                                  static_cast<unsigned long>(secsLeft));
            lv_label_set_text_static(ScreenLayoutLabelObject(1),
                                     "HR: Calibrating...");
            return true;
        }

        /* ---------------------------------------------------------------
         * RhythmFormer preprocessing, inference, and post-processing.
         * --------------------------------------------------------------- */
        const uint32_t pre_start = Get_SysTick_Cycle_Count32();
        if (!preProcess.DoPreProcess(g_cameraSnapshot, MIMAGE_RGB_BYTES)) {
            printf_err("Pre-processing failed.\n");
            return false;
        }
        const uint32_t pre_cycles = Get_SysTick_Cycle_Count32() - pre_start;

        /* Only surface LOW_LIGHT if this cycle hasn't already earned a
         * higher-precedence state (motion/no-face) above. */
        const bool lowLight = preProcess.WasLastFrameLowLight();
        if (lowLight && frameState == RhythmFormerState::kTracking) {
            frameState = RhythmFormerState::kLowLight;
        }

        if (!preProcess.IsChunkReady()) {
            if (frameState == RhythmFormerState::kTracking) {
                frameState = RhythmFormerState::kBuffering;
            }
            LogRhythmFormerState(frameState, result, faceState, lowLight);
            ScopedLVGLLock lv_lock;
            lv_label_set_text_fmt(ScreenLayoutLabelObject(0), "Frames: %zu", result.totalFrames);
            UpdateHrDisplay(result);
            return true;
        }

        const uint32_t inf_start = Get_SysTick_Cycle_Count32();
        if (!RunInference(model, profiler)) {
            printf_err("Inference failed.\n");
            preProcess.Reset();
            return false;
        }
        const uint32_t inf_cycles = Get_SysTick_Cycle_Count32() - inf_start;

        const uint32_t post_start = Get_SysTick_Cycle_Count32();
        if (!postProcess.DoPostProcess()) {
            printf_err("Post-processing failed.\n");
            return false;
        }
        const uint32_t post_cycles = Get_SysTick_Cycle_Count32() - post_start;

        /* Structured UART log for external UI tools: a live BPM readout and
         * waveform stream without linking against the library directly.
         * With stride=1, only bvp_new (= bvpRaw[last]) is genuinely new per
         * cycle, so a UI can append just that value instead of re-parsing
         * the full bvp_raw vector. */
        {
            const float* bvpRaw = result.bvpRaw;
            info("[RHYTHMFORMER_UI] frame=%zu chunk=%zu hr_bpm=%.3f hr_smoothed=%.3f "
                 "hr_raw=%.3f hr_conf=%.4f hr_valid=%d bvp_count=%zu bvp_new=%+.6f "
                 "bvp_raw=%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                 result.totalFrames,
                 result.chunkIndex,
                 static_cast<double>(result.hrBpm),
                 static_cast<double>(postProcess.GetSmoothedBpm()),
                 static_cast<double>(result.rawHrBpm),
                 static_cast<double>(result.hrConfidence),
                 result.hrValid ? 1 : 0,
                 postProcess.GetBvpHistoryCount(),
                 static_cast<double>(bvpRaw[kRhythmFormerFrameDepth - 1]),
                 static_cast<double>(bvpRaw[0]), static_cast<double>(bvpRaw[1]),
                 static_cast<double>(bvpRaw[2]), static_cast<double>(bvpRaw[3]),
                 static_cast<double>(bvpRaw[4]), static_cast<double>(bvpRaw[5]),
                 static_cast<double>(bvpRaw[6]), static_cast<double>(bvpRaw[7]),
                 static_cast<double>(bvpRaw[8]), static_cast<double>(bvpRaw[9]));

            /* Periodic full-waveform snapshot (~once/second) so a UI that
             * (re)connects mid-session can populate the whole waveform at
             * once instead of accumulating it point by point. One buffered
             * info() call per dump avoids interleaving with other UART
             * output. */
            constexpr size_t kUiHistoryDumpIntervalChunks = 30;
            if ((result.chunkIndex % kUiHistoryDumpIntervalChunks) == 0) {
                const size_t histCount = postProcess.GetBvpHistoryCount();
                const float* hist      = postProcess.GetBvpHistory();

                static char s_bvpDumpBuf[64 + kBvpHistoryLen * 10];
                size_t off = static_cast<size_t>(snprintf(
                    s_bvpDumpBuf, sizeof(s_bvpDumpBuf),
                    "[RHYTHMFORMER_UI_BVP] chunk=%zu bvp_count=%zu bvp_hist=",
                    result.chunkIndex, histCount));

                for (size_t i = 0; i < histCount && off < sizeof(s_bvpDumpBuf) - 1; ++i) {
                    off += static_cast<size_t>(snprintf(
                        s_bvpDumpBuf + off, sizeof(s_bvpDumpBuf) - off,
                        (i + 1 < histCount) ? "%.6f," : "%.6f",
                        static_cast<double>(hist[i])));
                }

                info("%s\n", s_bvpDumpBuf);
            }
        }

        LogRhythmFormerState(frameState, result, faceState, lowLight);

        {
            ScopedLVGLLock lv_lock;
            lv_label_set_text_fmt(ScreenLayoutLabelObject(0),
                                  "Frames: %zu  Chunk: %zu",
                                  result.totalFrames,
                                  result.chunkIndex);
            UpdateHrDisplay(result);

#if SHOW_INF_TIME
            const double inf_ms  = ((double)inf_cycles / SystemCoreClock) * 1000.0;
            const double inf_fps = (double)SystemCoreClock / (inf_cycles > 0 ? inf_cycles : 1);
            lv_label_set_text_fmt(ScreenLayoutLabelObject(2), "Inf time: %.2f ms", inf_ms);
            lv_label_set_text_fmt(ScreenLayoutLabelObject(3), "Inf rate: %.2f /s", inf_fps);
#endif
#if SHOW_PIPELINE_TIME
            const uint32_t pipeline_cycles = pre_cycles + inf_cycles + post_cycles;
            const double pipe_ms = ((double)pipeline_cycles / SystemCoreClock) * 1000.0;
            lv_label_set_text_fmt(ScreenLayoutLabelObject(4), "Pipeline: %.2f ms", pipe_ms);
#endif
        }

        return true;
    }

} /* namespace app */
} /* namespace alif */