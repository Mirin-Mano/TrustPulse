#include "mlek/fwk/tflm/FomoFaceModel.hpp"
#include "mlek/log/log_macros.h"

const tflite::MicroOpResolver& arm::app::fwk::tflm::FomoFaceModel::GetOpResolver()
{
    return this->m_opResolver;
}

bool arm::app::fwk::tflm::FomoFaceModel::EnlistOperations()
{
    /* The Vela compiler folds all FOMO ops into a single ethos-u custom op.
     * No DEQUANTIZE or NMS node is left on the CPU (unlike the UltraFace
     * slim model), so only the NPU delegate needs to be registered. */
    if (kTfLiteOk == this->m_opResolver.AddEthosU()) {
        info("FomoFace: added %s support to op resolver\n", tflite::GetString_ETHOSU());
    } else {
        printf_err("FomoFace: failed to add Arm NPU support to op resolver.");
        return false;
    }
    return true;
}
