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
