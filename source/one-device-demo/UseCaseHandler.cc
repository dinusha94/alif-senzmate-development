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

/**************************************************************************//**
 * @author   Dinusha Nuwan
 * @email    dinusha@senzmate.com
 * @version  V1.0.0
 * @date     22-11-2024
 * @brief    None. 
 * @bug      None.
 * @Note     Customized to use with live microphone data
 ******************************************************************************/

#include "UseCaseHandler.hpp"

#include "YoloFastestModel.hpp"
#include "UseCaseCommonUtils.hpp"
#include "DetectorPostProcessing.hpp"
#include "DetectorPreProcessing.hpp"
#include "ScreenLayout.hpp"
#include "hal.h"
#include "log_macros.h"

#include <cinttypes>
#include <cmath>
#include <vector>
#include <cstdint>
#include <cstring>
#include "delay.h"
#include "lvgl.h"
#include "lv_port.h"
#include "lv_paint_utils.h"

#include "Classifier.hpp"
#include "MobileNetModel.hpp"
#include "ImageUtils.hpp"
#include "ImgClassProcessing.hpp"

#include "FaceEmbedding.hpp" 

#include "AsrClassifier.hpp"
#include "AsrResult.hpp"
#include "AudioUtils.hpp"
#include "OutputDecode.hpp"
#include "Wav2LetterModel.hpp"
#include "Wav2LetterPostprocess.hpp"
#include "Wav2LetterPreprocess.hpp"

#include "ImageCrop.hpp"

#define AUDIO_SAMPLES_KWS 32000 
static int16_t audio_inf_kws[AUDIO_SAMPLES_KWS];

#define MIMAGE_X 224
#define MIMAGE_Y 224

#define LIMAGE_X        192
#define LIMAGE_Y        192
#define LV_ZOOM         (2 * 256)

namespace {
lv_style_t boxStyle;
lv_color_t  lvgl_image[LIMAGE_Y][LIMAGE_X] __attribute__((section(".bss.lcd_image_buf")));                      // 192x192x2 = 73,728
};

static uint8_t black_image_data[LIMAGE_X * LIMAGE_Y * 3]
    __attribute__((section(".bss.black_image")));

using arm::app::Profiler;
using arm::app::ApplicationContext;
using arm::app::Model;
using arm::app::YoloFastestModel;
using arm::app::DetectorPreProcess;
using arm::app::DetectorPostProcess;
using arm::app::ImgClassPreProcess; 


namespace alif {
namespace app {

    using namespace arm::app;

    namespace object_detection {
    using namespace arm::app::object_detection;

    }



    /* Print the output tensor from the model */
    void PrintTfLiteTensor(TfLiteTensor* tensor) {
        if (tensor == nullptr) {
            info("Tensor is null \n");
            return;
        }

        // Check if the tensor is of type int8
        if (tensor->type != kTfLiteInt8) {
            info("Tensor is not of type int8! Got type: %d\n", tensor->type);
            return;
        }

        // Get the number of elements in the tensor
        int numElements = 1;
        for (int i = 0; i < tensor->dims->size; ++i) {
            numElements *= tensor->dims->data[i];
        }

        // Cast the tensor's data pointer to int8
        int8_t* data = tensor->data.int8;

        // Print the tensor data
        info("Tensor contents: \n");
        for (int i = 0; i < numElements; ++i) {
            info("Element %d: %d\n", i, data[i]);  // %d is for printing int8 values
        }
    }

    static void DeleteBoxes(lv_obj_t *frame)
    {
        // Assume that child 0 of the frame is the image itself
        int children = lv_obj_get_child_cnt(frame);
        while (children > 1) {
            lv_obj_del(lv_obj_get_child(frame, 1));
            children--;
        }
    }

    // A helper function to crop the image based on the detection box.
    bool CropDetectedObject(const uint8_t* currImage, int inputImgCols, int inputImgRows, const object_detection::DetectionResult& result, uint8_t* croppedImage) {
        // Ensure the bounding box coordinates are within the image dimensions
        int x0 = result.m_x0;
        int y0 = result.m_y0;
        int w = result.m_w;
        int h = result.m_h;

        if (x0 < 0 || y0 < 0 || (x0 + w) > inputImgCols || (y0 + h) > inputImgRows) {
            printf_err("Invalid detection box coordinates for cropping.\n");
            return false;
        }

        // Crop the image by copying pixels from the detected region
        for (int y = 0; y < h; ++y) {
            int sourceOffset = ((y0 + y) * inputImgCols + x0) * 3; // Assuming 3 channels (RGB) per pixel
            int destOffset = (y * w) * 3;

            // Copy the row of the detection box from the original image to the cropped image
            std::memcpy(&croppedImage[destOffset], &currImage[sourceOffset], w * 3);
        }

        return true;
    }

    // This is the function that processes all detection results and crops the corresponding regions.
    bool ProcessDetectionsAndCrop(const uint8_t* currImage, int inputImgCols, int inputImgRows, const std::vector<object_detection::DetectionResult>& results, arm::app::ApplicationContext& context) {

        // auto croppedImages = context.Get<std::shared_ptr<std::vector<std::vector<uint8_t>>>>("cropped_images");
        auto croppedImages = context.Get<std::shared_ptr<std::vector<CroppedImageData>>>("cropped_images");
        bool faceDetected = false;

        if (!croppedImages) {
            printf_err("Failed to retrieve cropped_images from context.\n");
            return false;
        }
        
        for (const auto& result: results) {

            // Calculate size of the cropped image based on the detection box dimensions
            int croppedWidth = result.m_w;
            int croppedHeight = result.m_h;

            // Allocate memory for the cropped image (assuming RGB format, hence *3 for channels)
            std::vector<uint8_t> croppedImage(croppedWidth * croppedHeight * 3);

            if (!croppedImage.data()) {
                return false;
            }

            // Crop the detected object from the current image
            if (CropDetectedObject(currImage, inputImgCols, inputImgRows, result, croppedImage.data())) {
   
                croppedImages->emplace_back(CroppedImageData{ std::move(croppedImage), croppedWidth, croppedHeight });
                faceDetected = true;

                 if (faceDetected) {
                    context.Set<bool>("face_detected_flag", true);  // Set flag to true when object is detected
                    info("Face detected, face flag set exitiong loop ..\n");
                    break; // exit from the for loop
                }

            } else {
                info("Failed to crop detected object at {x=%d, y=%d, w=%d, h=%d}\n", result.m_x0, result.m_y0, result.m_w, result.m_h);
            }
        }

        return true;
    }

    /* Function to process cropped faces to get the embedding vector */
    bool ClassifyImageHandler(ApplicationContext& ctx, int32_t mode) {

        auto& profiler = ctx.Get<Profiler&>("profiler");
        auto& model = ctx.Get<Model&>("recog_model");

        // Retrieve the name 
        // auto my_name = ctx.Get<std::string>("my_name");
        std::string my_name;  
        SimilarityResult identified_person;

        if (mode == 0){
            // Retrieve the name 
            my_name = ctx.Get<std::string>("my_name");
        }else if (mode == 1)
        {
            my_name = "inference";
        }

        // Retrieve the cropped_images vector from the context
        auto croppedImages = ctx.Get<std::shared_ptr<std::vector<CroppedImageData>>>("cropped_images");
        
        // Check if the pointer is valid
        if (!croppedImages) {
            printf_err("Failed to retrieve cropped_images from context.\n");
            return false;
        }

        // Retrieve the face embedding collection
        auto& embeddingCollection = ctx.Get<FaceEmbeddingCollection&>("face_embedding_collection");

        if (!model.IsInited()) {
            printf_err("Model is not initialised! Terminating processing.\n");
            return false;
        }

        const uint32_t nCols       = MIMAGE_X;
        const uint32_t nRows       = MIMAGE_Y;

        // Process the current set of cropped images
        for (const auto& croppedImageData: *croppedImages) {
            // Access the image, width, and height
            const std::vector<uint8_t>& image = croppedImageData.image;
            int width = croppedImageData.width;
            int height = croppedImageData.height;

            // Allocate memory for the destination image
            uint8_t *dstImage = (uint8_t *)malloc(nCols * nRows * 3);
            if (!dstImage) {
                perror("Failed to allocate memory for destination image");
                return false;
            }

            // preprocessing for embedding model (MobileNet v2)
            crop_and_interpolate_(const_cast<uint8_t*>(image.data()), 
                                            width, height,
                                            dstImage, 
                                            nCols, nRows, 
                                            3 * 8);

            // Do inference
            TfLiteTensor* inputTensor = model.GetInputTensor(0);
            TfLiteTensor* outputTensor = model.GetOutputTensor(0);

            if (!inputTensor->dims) {
                printf_err("Invalid input tensor dims\n");
                return false;
            } else if (inputTensor->dims->size < 4) {
                printf_err("Input tensor dimension should be = 4\n");
                return false;
            }

            // /* Set up pre and post-processing. */
            ImgClassPreProcess preProcess = ImgClassPreProcess(inputTensor, model.IsDataSigned());

            const size_t imgSz = inputTensor->bytes;

            /* Run the pre-processing, inference and post-processing. */
            if (!preProcess.DoPreProcess(dstImage, imgSz)) {
                printf_err("Pre-processing failed.");
                return false;
            }
            
            if (!RunInference(model, profiler)) {
                printf_err("Inference failed.");
                return false;
            }

            // Convert the output tensor to a vector of int8
            std::vector<int8_t> int8_feature_vector(outputTensor->data.int8, 
                                                    outputTensor->data.int8 + outputTensor->bytes);

            // Save the feature vector along with the name in the embedding collection
            // embeddingCollection.AddEmbedding(my_name, int8_feature_vector);

            if (mode == 0){
                // Save the feature vector along with the name in the embedding collection
                embeddingCollection.AddEmbedding(my_name, int8_feature_vector);
            }else if (mode == 1)
            {
                // std::string mostSimilarPerson = embeddingCollection.FindMostSimilarEmbedding(int8_feature_vector);
                // ctx.Set<std::string>("person_id", mostSimilarPerson);

                identified_person = embeddingCollection.FindMostSimilarEmbedding(int8_feature_vector);
                ctx.Set<std::string>("person_id", identified_person.name);
            }

            free(dstImage);

        }

        // Clear the cropped images after processing to prepare for the next set
        if (croppedImages) {
            croppedImages->clear(); // Clear the vector of cropped images
        } else {
            printf_err("Failed to retrieve cropped_images from context.\n");
        }

        if (mode == 0){
            {
            ScopedLVGLLock lv_lock;
            lv_label_set_text_fmt(ScreenLayoutLabelObject(2), "Pose Now !! ");

            } // ScopedLVGLLock
        }else if (mode == 1)
        {
           {
            ScopedLVGLLock lv_lock;
            lv_label_set_text_fmt(ScreenLayoutLabelObject(2), "Recognition score : %.3f", identified_person.similarity);

            } // ScopedLVGLLock
        }

        return true;
    }



    bool ObjectDetectionInit()
    {

        ScreenLayoutInit(lvgl_image, sizeof lvgl_image, LIMAGE_X, LIMAGE_Y, LV_ZOOM);
        uint32_t lv_lock_state = lv_port_lock();
        lv_label_set_text_static(ScreenLayoutHeaderObject(), "Person registration App");
        lv_label_set_text_static(ScreenLayoutLabelObject(0), "Faces Detected: 0");
        lv_label_set_text_static(ScreenLayoutLabelObject(1), "Registered: 0");
        lv_label_set_text_static(ScreenLayoutLabelObject(2), "");

        lv_style_init(&boxStyle);
        lv_style_set_bg_opa(&boxStyle, LV_OPA_TRANSP);
        lv_style_set_pad_all(&boxStyle, 0);
        lv_style_set_border_width(&boxStyle, 0);
        lv_style_set_outline_width(&boxStyle, 2);
        lv_style_set_outline_pad(&boxStyle, 0);
        lv_style_set_outline_color(&boxStyle, lv_theme_get_color_primary(ScreenLayoutHeaderObject()));
        lv_style_set_radius(&boxStyle, 4);
        lv_port_unlock(lv_lock_state);

        /* Initialise the camera */
        int err = hal_image_init();
        if (0 != err) {
            printf_err("hal_image_init failed with error: %d\n", err);
        }

        return true;
    }

    static void ReplaceImageWithBlack()
    {
        const uint8_t* ptr = black_image_data;
        // Write the black buffer to the screen
        write_to_lvgl_buf(LIMAGE_Y, LIMAGE_X, ptr, &lvgl_image[0][0]);
        // Invalidate the image object to refresh the display
        lv_obj_invalidate(ScreenLayoutImageObject());
        lv_label_set_text_static(ScreenLayoutHeaderObject(), "State you're Name");
        sleep_or_wait_msec(10);
    }

    /**
     * @brief           Draw boxes directly on the LCD for all detected objects.
     * @param[in]       results            Vector of detection results to be displayed.
     **/
    static void DrawDetectionBoxes(
           const std::vector<object_detection::DetectionResult>& results,
           int imgInputCols, int imgInputRows);


    /* Object detection inference handler. */
    bool ObjectDetectionHandler(ApplicationContext& ctx, int32_t mode)
    {
        auto& profiler = ctx.Get<Profiler&>("profiler");
        auto& model = ctx.Get<Model&>("det_model");

        // Retrieve the name 
        // auto my_name = ctx.Get<std::string>("my_name");

        std::string my_name;  

        if (mode == 0){
            // Retrieve the name 
            my_name = ctx.Get<std::string>("my_name");
        }else if (mode == 1)
        {
            my_name = "inference";
        }

        if (!model.IsInited()) {
            printf_err("Model is not initialised! Terminating processing.\n");
            return false;
        }

        TfLiteTensor* inputTensor = model.GetInputTensor(0);
        TfLiteTensor* outputTensor0 = model.GetOutputTensor(0);
        TfLiteTensor* outputTensor1 = model.GetOutputTensor(1);

        if (!inputTensor->dims) {
            printf_err("Invalid input tensor dims\n");
            return false;
        } else if (inputTensor->dims->size < 3) {
            printf_err("Input tensor dimension should be >= 3\n");
            return false;
        }

        TfLiteIntArray* inputShape = model.GetInputShape(0);

        const int inputImgCols = inputShape->data[YoloFastestModel::ms_inputColsIdx];
        const int inputImgRows = inputShape->data[YoloFastestModel::ms_inputRowsIdx];

        /* Set up pre and post-processing. */
        DetectorPreProcess preProcess = DetectorPreProcess(inputTensor, true, model.IsDataSigned());

        std::vector<object_detection::DetectionResult> results;
        const object_detection::PostProcessParams postProcessParams {
            inputImgRows, inputImgCols, object_detection::originalImageSize,
            object_detection::anchor1, object_detection::anchor2
        };
        DetectorPostProcess postProcess = DetectorPostProcess(outputTensor0, outputTensor1,
                results, postProcessParams);

        /* Ensure there are no results leftover from previous inference when running all. */
        results.clear();

        const uint8_t* currImage = hal_get_image_data(inputImgCols, inputImgRows);
        if (!currImage) {
            printf_err("hal_get_image_data failed");
            return false;
        }

        // Display and inference start
        {
            ScopedLVGLLock lv_lock;

            write_to_lvgl_buf(inputImgCols, inputImgRows,
                            currImage, &lvgl_image[0][0]);
            lv_obj_invalidate(ScreenLayoutImageObject());

            lv_led_on(ScreenLayoutLEDObject());

            const size_t copySz = inputTensor->bytes;

            if (!my_name.empty()) {  // ctx.Get<bool>("buttonflag") ||

            info(" DETECTION  ...............\n");

            /* Run the pre-processing, inference and post-processing. */
            if (!preProcess.DoPreProcess(currImage, copySz)) {
                printf_err("Pre-processing failed.");
                return false;
            }

            /* Run inference over this image. */

            if (!RunInference(model, profiler)) {
                printf_err("Inference failed.");
                return false;
            }

            if (!postProcess.DoPostProcess()) {
                printf_err("Post-processing failed.");
                return false;
            }

            if (!ProcessDetectionsAndCrop(currImage, inputImgCols, inputImgRows, results, ctx)){
                printf_err("Cropping failed.");
                return false;
            }

            // lv_label_set_text_fmt(ScreenLayoutHeaderObject(), "Face Detection");
            lv_label_set_text_fmt(ScreenLayoutLabelObject(0), "Faces Detected: %i", results.size());

            // if (ctx.Get<bool>("face_detected_flag")) {
            //     lv_label_set_text_fmt(ScreenLayoutLabelObject(1), "Registered: %s", my_name.c_str()); // display the registered person name
            // }else {
            //     lv_label_set_text_fmt(ScreenLayoutLabelObject(1), "Registered: 0");
            // }

            if (mode == 0){
                lv_label_set_text_fmt(ScreenLayoutHeaderObject(), "MODE: Registration");
            
                if (ctx.Get<bool>("face_detected_flag")) {
                    lv_label_set_text_fmt(ScreenLayoutLabelObject(1), "Registered: %s", my_name.c_str()); // display the registered person name
                }else {
                    // lv_label_set_text_fmt(ScreenLayoutLabelObject(1), "");
                    lv_label_set_text(ScreenLayoutLabelObject(1), "");
                }

            }else if (mode == 1)
            {
                lv_label_set_text_fmt(ScreenLayoutHeaderObject(), "MODE: Inference");
                auto whoAmI = ctx.Get<std::string>("person_id");  // retrieve the person ID
            lv_label_set_text_fmt(ScreenLayoutLabelObject(1), "Name : %s", whoAmI.c_str());
            }


            // ctx.Set<bool>("buttonflag", false);

            }

            //  lv_label_set_text_fmt(ScreenLayoutLabelObject(2), "");
             lv_label_set_text(ScreenLayoutLabelObject(2), "");

            /* Draw boxes. */
            DrawDetectionBoxes(results, inputImgCols, inputImgRows);

        } // ScopedLVGLLock

        return true;
    }

    static void CreateBox(lv_obj_t *frame, int x0, int y0, int w, int h)
    {
        lv_obj_t *box = lv_obj_create(frame);
        lv_obj_set_size(box, w, h);
        lv_obj_add_style(box, &boxStyle, LV_PART_MAIN);
        lv_obj_set_pos(box, x0, y0);
    }

    static void DrawDetectionBoxes(const std::vector<object_detection::DetectionResult>& results,
                                   int imgInputCols, int imgInputRows)
    {
        lv_obj_t *frame = ScreenLayoutImageHolderObject();
        float xScale = (float) lv_obj_get_content_width(frame) / imgInputCols;
        float yScale = (float) lv_obj_get_content_height(frame) / imgInputRows;

        DeleteBoxes(frame);

        for (const auto& result: results) {
            CreateBox(frame,
                      floor(result.m_x0 * xScale),
                      floor(result.m_y0 * yScale),
                      ceil(result.m_w * xScale),
                      ceil(result.m_h * yScale));
        }
    }

    /* ASR inference handler. */
    std::string ClassifyAudioHandler(ApplicationContext& ctx, uint32_t mode, bool runAll)
    {
        auto& model          = ctx.Get<Model&>("asr_model");
        auto& profiler       = ctx.Get<Profiler&>("profiler");
        auto mfccFrameLen    = ctx.Get<uint32_t>("frameLength");
        auto mfccFrameStride = ctx.Get<uint32_t>("frameStride");
        auto scoreThreshold  = ctx.Get<float>("scoreThreshold");
        auto inputCtxLen     = ctx.Get<uint32_t>("ctxLen");     

        if (!model.IsInited()) {
            printf_err("Model is not initialised! Terminating processing.\n");
            return "false";
        }

        TfLiteTensor* inputTensor  = model.GetInputTensor(0);
        TfLiteTensor* outputTensor = model.GetOutputTensor(0);

        /* Get input shape. Dimensions of the tensor should have been verified by
         * the callee. */
        TfLiteIntArray* inputShape = model.GetInputShape(0);

        const uint32_t inputRowsSize = inputShape->data[Wav2LetterModel::ms_inputRowsIdx];
        const uint32_t inputInnerLen = inputRowsSize - (2 * inputCtxLen);

        /* Audio data stride corresponds to inputInnerLen feature vectors. */
        const uint32_t audioDataWindowLen = (inputRowsSize - 1) * mfccFrameStride + (mfccFrameLen);
        const uint32_t audioDataWindowStride = inputInnerLen * mfccFrameStride;

        /* NOTE: This is only used for time stamp calculation. */
        const float secondsPerSample = (1.0 / audio::Wav2LetterMFCC::ms_defaultSamplingFreq);

        /* Set up pre and post-processing objects. */
        AsrPreProcess preProcess = AsrPreProcess(inputTensor,
                                                 Wav2LetterModel::ms_numMfccFeatures,
                                                 inputShape->data[Wav2LetterModel::ms_inputRowsIdx],
                                                 mfccFrameLen,
                                                 mfccFrameStride);

        std::vector<ClassificationResult> singleInfResult;
        const uint32_t outputCtxLen = AsrPostProcess::GetOutputContextLen(model, inputCtxLen);
        AsrPostProcess postProcess  = AsrPostProcess(outputTensor,
                                                    ctx.Get<AsrClassifier&>("classifier"),
                                                    ctx.Get<std::vector<std::string>&>("labels"),
                                                    singleInfResult,
                                                    outputCtxLen,
                                                    Wav2LetterModel::ms_blankTokenIdx,
                                                    Wav2LetterModel::ms_outputRowsIdx);

        // Retrieve the audio_inf pointer from the context
        // auto audio_inf_vector = ctx.Get<std::vector<int16_t>>("audio_inf_vector");
        // // const int16_t* audio_inf = audio_inf_vector.data(); 

        /* make screen black (better than sucked image) */
        {
            ScopedLVGLLock lv_lock;
            ReplaceImageWithBlack();
            // lv_label_set_text_fmt(ScreenLayoutLabelObject(1), "");
            lv_label_set_text(ScreenLayoutLabelObject(1), "");
        }

        uint32_t audioArrSize = AUDIO_SAMPLES_KWS; // 16000 + 8000;

        static bool audio_inited;
        std::string finalResultStr;

        if (!audio_inited) {
            int err = hal_audio_init(16000);  // Initialize audio at 16,000 Hz
            if (err) {
                info("hal_audio_init failed with error: %d\n", err);
            }
            audio_inited = true;
        }
       
        /* Loop to process audio clips. */
        do {
           
            /* Get the current audio buffer and respective size. */
            hal_get_audio_data(audio_inf_kws, AUDIO_SAMPLES_KWS); // recorded audio data in mono

            // Wait until the buffer is fully populated
            int err = hal_wait_for_audio();
            if (err) {
                info("hal_wait_for_audio failed with error: %d\n", err);
            }

            hal_audio_preprocessing(audio_inf_kws, AUDIO_SAMPLES_KWS);             

            /* Audio clip needs enough samples to produce at least 1 MFCC feature. */
            if (audioArrSize < mfccFrameLen) {
                info("Not enough audio samples, minimum needed is %" PRIu32 "\n",
                           mfccFrameLen);
                return "false";
            }

            /* Creating a sliding window through the whole audio clip. */
            auto audioDataSlider = audio::FractionalSlidingWindow<const int16_t>(
                audio_inf_kws, audioArrSize, audioDataWindowLen, audioDataWindowStride);

            /* Declare a container for final results. */
            std::vector<asr::AsrResult> finalResults;

            size_t inferenceWindowLen = audioDataWindowLen;

            /* Start sliding through audio clip. */
            while (audioDataSlider.HasNext()) {

                /* If not enough audio, see how much can be sent for processing. */
                size_t nextStartIndex = audioDataSlider.NextWindowStartIndex();
                if (nextStartIndex + audioDataWindowLen > audioArrSize) {
                    inferenceWindowLen = audioArrSize - nextStartIndex;
                }

                const int16_t* inferenceWindow = audioDataSlider.Next();

                info("Inference %zu/%zu\n",
                     audioDataSlider.Index() + 1,
                     static_cast<size_t>(ceilf(audioDataSlider.FractionalTotalStrides() + 1)));

                /* Run the pre-processing, inference and post-processing. */
                if (!preProcess.DoPreProcess(inferenceWindow, inferenceWindowLen)) {
                    printf_err("Pre-processing failed.");
                    return "false";
                }

                if (!RunInference(model, profiler)) {
                    printf_err("Inference failed.");
                    return "false";
                }

                /* Post processing needs to know if we are on the last audio window. */
                postProcess.m_lastIteration = !audioDataSlider.HasNext();
                if (!postProcess.DoPostProcess()) {
                    printf_err("Post-processing failed.");
                    return "false";
                }

                /* Add results from this window to our final results vector. */
                finalResults.emplace_back(asr::AsrResult(
                    singleInfResult,
                    (audioDataSlider.Index() * secondsPerSample * audioDataWindowStride),
                    audioDataSlider.Index(),
                    scoreThreshold));

            } /* while (audioDataSlider.HasNext()) */

            ctx.Set<std::vector<asr::AsrResult>>("results", finalResults);

            std::vector<ClassificationResult> combinedResults;
            for (const auto& result : finalResults) {
                combinedResults.insert(
                    combinedResults.end(), result.m_resultVec.begin(), result.m_resultVec.end());
            }

            /* Get the decoded result for the combined result. */
            finalResultStr = audio::asr::DecodeOutput(combinedResults);


            switch (mode)
            {
                case 0:
                    info("Complete recognition: %s\n", finalResultStr.c_str());
                    // Check if the result contains "Hi"
                    if (finalResultStr.find("go") != std::string::npos) {
                        info("The word 'Hi' was detected in the recognition result.");
                        ctx.Set<bool>("kw_flag", true);
                    }
                    break;
                
                case 1:
                    info("Complete recognition: %s\n", finalResultStr.c_str());
                    // send_name(finalResultStr);
                    // ctx.Set<std::string&>("my_name", finalResultStr);
                    ctx.Set<std::string>("my_name", finalResultStr);

                    break;
                
                default:
                    break;
            }

          
        } while (runAll); 

        return finalResultStr;
    }

} /* namespace app */
} /* namespace alif */