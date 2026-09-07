/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TSCAN_MODEL_HPP
#define TSCAN_MODEL_HPP

#include "mlek/fwk/tflm/TflmModel.hpp"

namespace arm::app::fwk::tflm {

class TscanModel : public TflmModel {

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

#endif /* TSCAN_MODEL_HPP */
