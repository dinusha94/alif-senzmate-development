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

#include "lvgl.h"
#include "lv_port.h"
#include "lv_paint_utils.h"

#include "Classifier.hpp"
#include "MobileNetModel.hpp"
#include "ImageUtils.hpp"
#include "ImgClassProcessing.hpp"

#include "FaceEmbedding.hpp" 
#include "ImageCrop.hpp"

#define MIMAGE_X 224
#define MIMAGE_Y 224

#define LIMAGE_X        192
#define LIMAGE_Y        192
#define LV_ZOOM         (2 * 256)

namespace {
lv_style_t boxStyle;
lv_color_t  lvgl_image[LIMAGE_Y][LIMAGE_X] __attribute__((section(".bss.lcd_image_buf")));                      // 192x192x2 = 73,728
};

using arm::app::Profiler;
using arm::app::ApplicationContext;
using arm::app::Model;
using arm::app::YoloFastestModel;
using arm::app::DetectorPreProcess;
using arm::app::DetectorPostProcess;
using arm::app::ImgClassPreProcess; 

namespace alif {
namespace app {

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
                  
                // Save the cropped image into the context
                croppedImages->emplace_back(CroppedImageData{ std::move(croppedImage), croppedWidth, croppedHeight });
                faceDetected = true;

                 if (faceDetected) {
                    context.Set<bool>("face_detected_flag", true);  // Set flag to true when object is detected
                    break; // exit from the for loop
                }

            } else {
                info("Failed to crop detected object at {x=%d, y=%d, w=%d, h=%d}\n", result.m_x0, result.m_y0, result.m_w, result.m_h);
            }
        }

        return true;
    }

    /* Function to process cropped faces to get the embedding vector */
    bool ClassifyImageHandler(ApplicationContext& ctx) {

        auto& profiler = ctx.Get<Profiler&>("profiler_class");
        auto& model = ctx.Get<Model&>("recog_model");

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
                printf_err("Failed to allocate memory for destination image");
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

            /* Set up pre and post-processing. */
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

            std::string mostSimilarPerson = embeddingCollection.FindMostSimilarEmbedding(int8_feature_vector);

            ctx.Set<std::string>("person_id", mostSimilarPerson);

            free(dstImage);

        }

        // Clear the cropped images after processing to prepare for the next set
        if (croppedImages) {
            croppedImages->clear(); // Clear the vector of cropped images
        } else {
            printf_err("Failed to retrieve cropped_images from context.\n");
        }

        return true;
    }



    bool ObjectDetectionInit()
    {

        ScreenLayoutInit(lvgl_image, sizeof lvgl_image, LIMAGE_X, LIMAGE_Y, LV_ZOOM);
        uint32_t lv_lock_state = lv_port_lock();
        lv_label_set_text_static(ScreenLayoutHeaderObject(), "Face Detection");
        lv_label_set_text_static(ScreenLayoutLabelObject(0), "Faces Detected: 0");
        lv_label_set_text_static(ScreenLayoutLabelObject(1), "Name : ");               // print the name
        lv_label_set_text_static(ScreenLayoutLabelObject(2), "192px image (24-bit)");

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



    /**
     * @brief           Presents inference results along using the data presentation
     *                  object.
     * @param[in]       results            Vector of detection results to be displayed.
     * @return          true if successful, false otherwise.
     **/
    static bool PresentInferenceResult(const std::vector<object_detection::DetectionResult>& results);

    /**
     * @brief           Draw boxes directly on the LCD for all detected objects.
     * @param[in]       results            Vector of detection results to be displayed.
     **/
    static void DrawDetectionBoxes(
           const std::vector<object_detection::DetectionResult>& results,
           int imgInputCols, int imgInputRows);


    /* Object detection inference handler. */
    bool ObjectDetectionHandler(ApplicationContext& ctx)
    {
        auto& profiler = ctx.Get<Profiler&>("profiler_det");
        auto& model = ctx.Get<Model&>("det_model");

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

            /* Display this image on the LCD. */
            write_to_lvgl_buf(inputImgCols, inputImgRows,
                            currImage, &lvgl_image[0][0]);
            lv_obj_invalidate(ScreenLayoutImageObject());

            if (!run_requested()) {
               lv_led_off(ScreenLayoutLEDObject());
               return false;
            }

            lv_led_on(ScreenLayoutLEDObject());

            const size_t copySz = inputTensor->bytes;


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

            lv_label_set_text_fmt(ScreenLayoutLabelObject(0), "Faces Detected: %i", results.size());
            auto whoAmI = ctx.Get<std::string>("person_id");  // retrieve the person ID
            lv_label_set_text_fmt(ScreenLayoutLabelObject(1), "Name : %s", whoAmI.c_str());

            /* Draw boxes. */
            DrawDetectionBoxes(results, inputImgCols, inputImgRows);

        } // ScopedLVGLLock

        return true;
    }








    static bool PresentInferenceResult(const std::vector<object_detection::DetectionResult>& results)
    {
        /* If profiling is enabled, and the time is valid. */
        info("Final results:\n");
        info("Total number of inferences: 1\n");

        for (uint32_t i = 0; i < results.size(); ++i) {
            info("%" PRIu32 ") (%f) -> %s {x=%d,y=%d,w=%d,h=%d}\n", i,
                results[i].m_normalisedVal, "Detection box:",
                results[i].m_x0, results[i].m_y0, results[i].m_w, results[i].m_h );
        }

        return true;
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

} /* namespace app */
} /* namespace alif */
