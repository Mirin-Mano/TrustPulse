#ifndef FOMO_FACE_DETECTOR_HPP
#define FOMO_FACE_DETECTOR_HPP

#include "mlek/fwk/iface/Model.hpp"

#include <cstddef>
#include <cstdint>

/* Default confidence threshold (0.8). Overridable at CMake configure time
 * via -Dalif_RhythmFormer_FACE_DETECT_CONF_THRESHOLD=<value>, which maps to
 * the compile-time definition FACE_DETECT_CONF_THRESHOLD. */
#if !defined(FACE_DETECT_CONF_THRESHOLD)
#define FACE_DETECT_CONF_THRESHOLD 0.8f
#endif

namespace arm {
namespace app {

/**
 * @brief   NPU-accelerated face presence detector using the FOMO 72×72 int8
 *          TFLite model (Vela-compiled).
 *
 *          Preprocessing:
 *            1. Nearest-neighbour resize: camera resolution → 72×72
 *            2. Per-pixel subtract 128: int8_val = (int16_t)pixel - 128
 *               (integer-only path — no floating-point math needed).
 *               Matches the model's input quantisation: scale ≈ 1/255,
 *               zero_point = -128.
 *
 *          The model output tensor has shape [1, 9, 9, 2] (int8):
 *            Channel 0: background probability (quantised)
 *            Channel 1: face probability       (quantised)
 *          FOMO grid stride is 8 px (72/8 = 9 cells per axis).
 *
 *          DetectFace() dequantises the output using the tensor's own
 *          quantisation parameters, then returns true if any grid cell's
 *          face probability ≥ confThreshold — a pure presence check
 *          without bounding-box output, matching the RhythmFormer use case's
 *          "is a face in frame?" requirement.
 *
 *          All methods are static; the class has no per-instance state.
 */
class FomoFaceDetector {
public:
    /** Input spatial dimensions (must match the compiled model). */
    static constexpr int kInputWidth    = 72;
    static constexpr int kInputHeight   = 72;
    static constexpr int kInputChannels = 3;

    /** Output grid dimensions: 72 / 8 (FOMO network stride) = 9. */
    static constexpr int kGridSize    = 9;

    /** Channel index for the face class in the [1, 9, 9, 2] output. */
    static constexpr int kFaceClassIdx = 1;

    /**
     * @brief   Run one FOMO inference pass and return whether a face is present.
     *
     * @param[in] model          Initialised FomoFaceModel (must be IsInited()).
     * @param[in] rgb888         Source frame in canonical RGB888 order (R first).
     * @param[in] width          Frame width in pixels.
     * @param[in] height         Frame height in pixels.
     * @param[in] confThreshold  Minimum dequantised face probability to count
     *                           as a detection (0–1, default 0.8).
     *
     * @return  true if at least one 9×9 grid cell has a dequantised face
     *          probability ≥ confThreshold.
     */
    static bool DetectFace(fwk::iface::Model& model,
                           const uint8_t*     rgb888,
                           size_t             width,
                           size_t             height,
                           float              confThreshold = FACE_DETECT_CONF_THRESHOLD);

private:
    /**
     * @brief   Nearest-neighbour resize + int8 quantise in one pass.
     *
     *          For each destination pixel (dx, dy) in the 72×72 grid:
     *            src_x = dx * srcWidth  / kInputWidth
     *            src_y = dy * srcHeight / kInputHeight
     *            int8_val = (int16_t)src_pixel - 128
     *
     *          Writes directly into the model's input tensor buffer
     *          (GetData<int8_t>()) to avoid an extra memcpy.
     */
    static void PreProcess(const uint8_t*     rgb888,
                           size_t             srcWidth,
                           size_t             srcHeight,
                           fwk::iface::Model& model);
};

} /* namespace app */
} /* namespace arm */

#endif /* FOMO_FACE_DETECTOR_HPP */
