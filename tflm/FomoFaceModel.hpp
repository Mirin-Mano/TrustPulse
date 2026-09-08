#ifndef FOMO_FACE_MODEL_HPP
#define FOMO_FACE_MODEL_HPP

#include "mlek/fwk/tflm/TflmModel.hpp"

namespace arm::app::fwk::tflm {

/**
 * @brief   TFLM model wrapper for the FOMO 72×72 int8 face-detection model
 *          (Vela-compiled for the Ethos-U NPU).
 *
 *          Model I/O (Vela-compiled):
 *            Input  : [1, 72, 72, 3]   int8
 *            Output : [1,  9,  9, 2]   int8
 *                     Channel 0 = background probability (quantised)
 *                     Channel 1 = face probability      (quantised)
 *
 *          Op set required: only the ethos-u custom op — Vela folds all
 *          compute into a single delegate node; no CPU DEQUANTIZE or NMS
 *          node is inserted at the output boundary (unlike UltraFace).
 *
 *          The op resolver size ms_maxOpCnt = 1 to keep the TFLM allocator
 *          table as tight as possible, following the same pattern as
 *          TscanModel / UltraFaceModel.
 */
class FomoFaceModel : public TflmModel {
public:
    /** Input tensor dimension indices (shape: [1, 72, 72, 3]). */
    static constexpr uint32_t ms_inputBatchIdx    = 0;
    static constexpr uint32_t ms_inputRowsIdx     = 1;
    static constexpr uint32_t ms_inputColsIdx     = 2;
    static constexpr uint32_t ms_inputChannelsIdx = 3;

    /** Expected spatial input size (must match the compiled model). */
    static constexpr int ms_inputWidth  = 72;
    static constexpr int ms_inputHeight = 72;

    /** Output grid size: 72 / 8 (FOMO network stride) = 9. */
    static constexpr int ms_gridSize = 9;

    /** Number of output classes: 0 = background, 1 = face. */
    static constexpr int ms_numClasses = 2;

protected:
    const tflite::MicroOpResolver& GetOpResolver() override;
    bool EnlistOperations() override;

private:
    /* ethos-u custom op only — all compute is on the NPU. */
    static constexpr int ms_maxOpCnt = 1;
    tflite::MicroMutableOpResolver<ms_maxOpCnt> m_opResolver;
};

} /* namespace arm::app::fwk::tflm */

#endif /* FOMO_FACE_MODEL_HPP */
