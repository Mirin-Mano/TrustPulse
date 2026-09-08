#ifndef RHYTHMFORMER_MODEL_HPP
#define RHYTHMFORMER_MODEL_HPP

#include "mlek/fwk/tflm/TflmModel.hpp"

namespace arm::app::fwk::tflm {

class RhythmFormerModel : public TflmModel {

public:
    static constexpr uint32_t ms_frameDepthIdx    = 0;
    static constexpr uint32_t ms_inputRowsIdx       = 1;
    static constexpr uint32_t ms_inputColsIdx       = 2;
    static constexpr uint32_t ms_inputChannelsIdx   = 3;

protected:
    const tflite::MicroOpResolver& GetOpResolver() override;
    bool EnlistOperations() override;

private:
    static constexpr int ms_maxOpCnt = 3;  /* PAD + DEQUANTIZE + EthosU */
    tflite::MicroMutableOpResolver<ms_maxOpCnt> m_opResolver;
};

} /* namespace arm::app::fwk::tflm */

#endif 
