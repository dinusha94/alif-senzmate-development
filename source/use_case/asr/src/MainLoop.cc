/*
 * SPDX-FileCopyrightText: Copyright 2021 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "hal.h"

#include "Labels.hpp"                /* For label strings. */
#include "UseCaseHandler.hpp"        /* Handlers for different user options. */
#include "Wav2LetterModel.hpp"       /* Model class for running inference. */
#include "UseCaseCommonUtils.hpp"    /* Utils functions. */
#include "AsrClassifier.hpp"         /* Classifier. */
#include "log_macros.h"             /* Logging functions */
#include "BufAttributes.hpp"        /* Buffer attributes to be applied */

#include "delay.h"
#include <iostream>
#include <cstring> 
#include <random>

namespace arm {
namespace app {
    static uint8_t  tensorArena[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;
    namespace asr {
        extern uint8_t* GetModelPointer();
        extern size_t GetModelLen();
    } /* namespace asr */
} /* namespace app */
} /* namespace arm */

bool last_btn1 = false; 

bool run_requested_(void)
{
    bool ret = false; // Default to no inference
    bool new_btn1;
    BOARD_BUTTON_STATE btn_state1;

    // Get the new button state (active low)
    BOARD_BUTTON1_GetState(&btn_state1);
    new_btn1 = (btn_state1 == BOARD_BUTTON_STATE_LOW); // true if button is pressed

    // Edge detector - run inference on the positive edge of the button pressed signal
    if (new_btn1 && !last_btn1) // Check for transition from not pressed to pressed
    {
        ret = true; // Inference requested
    }

    // Update the last button state
    last_btn1 = new_btn1;

    return ret; // Return whether inference should be run
}

/** @brief   Verify input and output tensor are of certain min dimensions. */
static bool VerifyTensorDimensions(const arm::app::Model& model);

void main_loop()
{
   
    init_trigger_tx();
    
    arm::app::Wav2LetterModel model;  /* Model wrapper object. */

    /* Load the model. */
    if (!model.Init(arm::app::tensorArena,
                    sizeof(arm::app::tensorArena),
                    arm::app::asr::GetModelPointer(),
                    arm::app::asr::GetModelLen())) {
        printf_err("Failed to initialise model\n");
        return;
    } else if (!VerifyTensorDimensions(model)) {
        printf_err("Model's input or output dimension verification failed\n");
        return;
    }

    /* Instantiate application context. */
    arm::app::ApplicationContext caseContext;
    std::vector <std::string> labels;
    GetLabelsVector(labels);
    arm::app::AsrClassifier classifier;  /* Classifier wrapper object. */

    arm::app::Profiler profiler{"asr"};
    caseContext.Set<arm::app::Profiler&>("profiler", profiler);
    caseContext.Set<arm::app::Model&>("model", model);
    caseContext.Set<uint32_t>("frameLength", arm::app::asr::g_FrameLength);
    caseContext.Set<uint32_t>("frameStride", arm::app::asr::g_FrameStride);
    caseContext.Set<float>("scoreThreshold", arm::app::asr::g_ScoreThreshold);  /* Score threshold. */
    caseContext.Set<uint32_t>("ctxLen", arm::app::asr::g_ctxLen);  /* Left and right context length (MFCC feat vectors). */
    caseContext.Set<const std::vector <std::string>&>("labels", labels);
    caseContext.Set<arm::app::AsrClassifier&>("classifier", classifier);

    bool executionSuccessful = true;
    
    while(1){
            
        // button press mode    
        if (run_requested_())
        {   
            executionSuccessful = ClassifyAudioHandler(
                                    caseContext,
                                    1,
                                    false);
                                    
            info(" recognition status : %d \n", executionSuccessful);
        }
        
    }

    info("Main loop terminated.\n");
}

static bool VerifyTensorDimensions(const arm::app::Model& model)
{
    /* Populate tensor related parameters. */
    TfLiteTensor* inputTensor = model.GetInputTensor(0);
    if (!inputTensor->dims) {
        printf_err("Invalid input tensor dims\n");
        return false;
    } else if (inputTensor->dims->size < 3) {
        printf_err("Input tensor dimension should be >= 3\n");
        return false;
    }

    TfLiteTensor* outputTensor = model.GetOutputTensor(0);
    if (!outputTensor->dims) {
        printf_err("Invalid output tensor dims\n");
        return false;
    } else if (outputTensor->dims->size < 3) {
        printf_err("Output tensor dimension should be >= 3\n");
        return false;
    }

    return true;
}