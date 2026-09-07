#ifndef RHYTHMFORMER_HANDLER_HPP
#define RHYTHMFORMER_HANDLER_HPP

#include "AppContext.hpp"
#include "mlek/fwk/tflm/RhythmFormerModel.hpp"

namespace alif {
namespace app {

    bool RhythmFormerHandlerInit(arm::app::fwk::tflm::RhythmFormerModel& model);
    bool RhythmFormerHandler(arm::app::ApplicationContext& ctx);

} /* namespace app */
} /* namespace alif */

#endif /* RHYTHMFORMER_HANDLER_HPP */
