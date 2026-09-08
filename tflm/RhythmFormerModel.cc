#include "mlek/fwk/tflm/RhythmFormerModel.hpp"
#include "mlek/log/log_macros.h"

const tflite::MicroOpResolver& arm::app::fwk::tflm::RhythmFormerModel::GetOpResolver()
{
    return this->m_opResolver;
}

bool arm::app::fwk::tflm::RhythmFormerModel::EnlistOperations()
{
    this->m_opResolver.AddPad();

    /* Vela leaves a DEQUANTIZE node on the CPU at the model output boundary.
     * Without this registration TFLM reports "Didn't find op for builtin
     * opcode 'DEQUANTIZE'" and tensor allocation fails. */
    this->m_opResolver.AddDequantize();

    if (kTfLiteOk == this->m_opResolver.AddEthosU()) {
        info("Added %s support to op resolver\n", tflite::GetString_ETHOSU());
    } else {
        printf_err("Failed to add Arm NPU support to op resolver.");
        return false;
    }
    return true;
}
