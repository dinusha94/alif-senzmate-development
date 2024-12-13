/* This file was ported to work on Alif Semiconductor Ensemble family of devices. */

/* Copyright (C) 2023 Alif Semiconductor - All Rights Reserved.
 * Use, distribution and modification of this code is permitted under the
 * terms stated in the Alif Semiconductor Software License Agreement
 *
 * You should have received a copy of the Alif Semiconductor Software
 * License Agreement with this file. If not, please write to:
 * contact@alifsemi.com, or visit: https://alifsemi.com/license
 *
 */

/*
 * Copyright (c) 2022 Arm Limited. All rights reserved.
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
#include "hal.h"                      /* Brings in platform definitions. */
#include "YoloFastestModel.hpp"       /* Model class for running inference. */
#include "UseCaseHandler.hpp"         /* Handlers for different user options. */
#include "UseCaseCommonUtils.hpp"     /* Utils functions. */
#include "log_macros.h"             /* Logging functions */
#include "BufAttributes.hpp"        /* Buffer attributes to be applied */
#include "MobileNetModel.hpp"       /* Model class for running inference. */
#include "delay.h"

#include "FaceEmbedding.hpp"        /* Class for face embedding related functions */
#include "Flash.hpp"                /* Class for external flash memory operations */
#include <iostream>
#include <cstring> 
#include <random>

#include "Labels.hpp"                /* For label strings. */
#include "Wav2LetterModel.hpp"       /* Model class for running inference. */
#include "AsrClassifier.hpp"         /* Classifier. */

namespace arm {
namespace app {
    
    namespace object_detection {
        extern uint8_t* GetModelPointer();
        extern size_t GetModelLen();
    } /* namespace object_detection */

    namespace img_class{
        extern uint8_t* GetModelPointer();
        extern size_t GetModelLen();
    } // namespace object_recognition

    namespace asr {
        extern uint8_t* GetModelPointer();
        extern size_t GetModelLen();
    } /* namespace asr */

    static uint8_t tensorArena[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;
} /* namespace app */
} /* namespace arm */

// Global variable to hold the received message
const int MAX_MESSAGE_LENGTH = 8;
char receivedMessage[MAX_MESSAGE_LENGTH];

/* callback function to handle name strings received from speech recognition process*/
void user_message_callback(char *message) {
    strncpy(receivedMessage, message, MAX_MESSAGE_LENGTH - 1);
    receivedMessage[MAX_MESSAGE_LENGTH - 1] = '\0'; // Ensure null-termination
    info("Message received in user callback: %s\n", message);
}

bool last_btn1 = false; 
bool last_btn2 = false; 

bool run_requested_btn_1(void)
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

bool run_requested_btn_2(void)
{
    bool ret = false; // Default to no inference
    bool new_btn2;
    BOARD_BUTTON_STATE btn_state2;

    // Get the new button state (active low)
    BOARD_BUTTON3_GetState(&btn_state2);
    new_btn2 = (btn_state2 == BOARD_BUTTON_STATE_LOW); // true if button is pressed

    // Edge detector - run inference on the positive edge of the button pressed signal
    if (new_btn2 && !last_btn2) // Check for transition from not pressed to pressed
    {
        ret = true; // Inference requested
    }

    // Update the last button state
    last_btn2 = new_btn2;

    return ret; // Return whether inference should be run
}

void main_loop()
{   
    /* Trigger when a name received from asr */
    // init_trigger_tx_custom(user_message_callback);

    arm::app::YoloFastestModel det_model;  /* Model wrapper object. */
    arm::app::MobileNetModel recog_model;
    arm::app::Wav2LetterModel asr_model;
    
    /* No need to initiate Classification since we use single camera*/
    if (!alif::app::ObjectDetectionInit()) {
        printf_err("Failed to initialise use case handler\n");
    }

    /* Load asr model */
    if (!asr_model.Init(arm::app::tensorArena,
                    sizeof(arm::app::tensorArena),
                    arm::app::asr::GetModelPointer(),
                    arm::app::asr::GetModelLen())) {
        printf_err("Failed to initialise model\n");
        return;
    } 

    /* Load the detection model. */
    if (!det_model.Init(arm::app::tensorArena,
                    sizeof(arm::app::tensorArena),
                    arm::app::object_detection::GetModelPointer(),
                    arm::app::object_detection::GetModelLen(),
                    asr_model.GetAllocator())) {
        printf_err("Failed to initialise model\n");
        return;
    }


    /* Load the recognition model. */
    if (!recog_model.Init(arm::app::tensorArena,
                    sizeof(arm::app::tensorArena),
                    arm::app::img_class::GetModelPointer(),
                    arm::app::img_class::GetModelLen(),
                    asr_model.GetAllocator())) {
        printf_err("Failed to initialise recognition model\n");
        return;
    }


    /* Instantiate application context. */
    arm::app::ApplicationContext caseContext;

    arm::app::Profiler profiler{"object_detection"};
    caseContext.Set<arm::app::Profiler&>("profiler", profiler);
    caseContext.Set<arm::app::Model&>("det_model", det_model);
    caseContext.Set<arm::app::Model&>("recog_model", recog_model);

  
    std::vector <std::string> labels;
    GetLabelsVector(labels);
    arm::app::AsrClassifier classifier;  /* Classifier wrapper object. */

    caseContext.Set<arm::app::Model&>("asr_model", asr_model);
    caseContext.Set<uint32_t>("frameLength", arm::app::asr::g_FrameLength);
    caseContext.Set<uint32_t>("frameStride", arm::app::asr::g_FrameStride);
    caseContext.Set<float>("scoreThreshold", arm::app::asr::g_ScoreThreshold);  /* Score threshold. */
    caseContext.Set<uint32_t>("ctxLen", arm::app::asr::g_ctxLen);  /* Left and right context length (MFCC feat vectors). */
    caseContext.Set<const std::vector <std::string>&>("labels", labels);
    caseContext.Set<arm::app::AsrClassifier&>("classifier", classifier);

    // Dynamically allocate the vector on the heap to hold CroppedImageData
    auto croppedImages = std::make_shared<std::vector<alif::app::CroppedImageData>>();
    caseContext.Set<std::shared_ptr<std::vector<alif::app::CroppedImageData>>>("cropped_images", croppedImages);

    // Set the context to save the facial embeddings and corresponding name
    FaceEmbeddingCollection faceEmbeddingCollection;
    caseContext.Set<FaceEmbeddingCollection&>("face_embedding_collection", faceEmbeddingCollection);

    // Collection to load stored face embeddings
    FaceEmbeddingCollection stored_collection;

    // flag to notify face detection
    bool faceFlag = false;
    caseContext.Set<bool>("face_detected_flag", faceFlag);

    // flag to notify button press event
    caseContext.Set<bool>("buttonflag", false);

    // Hardcoded name
    std::string myName = "";
    // caseContext.Set<std::string&>("my_name", myName);
    caseContext.Set<std::string>("my_name", "");

    std::string whoAmI = "";
    caseContext.Set<std::string>("person_id", whoAmI);

    bool avgEmbFlag = false;
    int loop_idx = 0;
    int32_t ret;

    int32_t mode = 0; // 0 - registration mode, 1 inference mode
    int32_t last_mode = 0;

    while(1) {

        alif::app::ObjectDetectionHandler(caseContext, mode);

        // Button press mode    
        if (run_requested_btn_1())
        {   
            mode = 0;
            myName = alif::app::ClassifyAudioHandler(
                                    caseContext,
                                    1,
                                    false);
                                    
            info("recognition Name : %s \n", myName.c_str());
        }

        // switch to inference mode
        if (run_requested_btn_2())
        {   
            mode = 1;
            info("swithing to inference mode \n");
            continue;
        }

        


        /* extract the facial embedding and register the person */
        if (mode == 0){
            if (caseContext.Get<bool>("face_detected_flag") && !myName.empty()) { 
                avgEmbFlag = true;
                info("registration .. \n");

                if (avgEmbFlag && (loop_idx < 5)){
                    info("Averaging embeddings .. \n");
                    alif::app::ClassifyImageHandler(caseContext, mode); 
                    sleep_or_wait_msec(1000); /* wait for possible pose changes */
                    loop_idx ++; 
                }else {
                    avgEmbFlag = false;
                    loop_idx = 0;

                    // average the embedding fro the myName
                    faceEmbeddingCollection.CalculateAverageEmbeddingAndSave(myName);
                    info("Averaging finished and saved .. \n");

                    /* save embedding data to external flash  */
                    ret = flash_send(faceEmbeddingCollection);
                    /* TODO: investigate this issue */
                    ospi_flash_read_dummy();

                    caseContext.Set<bool>("face_detected_flag", false); // Reset flag 
                    myName.clear();

                    caseContext.Set<std::string>("my_name", myName);

                }
            }
        }
        else if (mode == 1)
        {
            if (last_mode != mode){
                // retrieve the person registration data
                ret = ospi_flash_read_collection(stored_collection);
                // stored_collection.PrintEmbeddings();
                /* Save the face embedding collection in context */
                caseContext.Set<FaceEmbeddingCollection&>("recorded_face_embedding_collection", stored_collection);

                info("data loaded correctly \n");
            }
            alif::app::ClassifyImageHandler(caseContext, mode);  // Run feature extraction
            sleep_or_wait_msec(100); // keep some delay here
            
            
        } // end inference
 
    last_mode = mode;       
        
    }
    
}