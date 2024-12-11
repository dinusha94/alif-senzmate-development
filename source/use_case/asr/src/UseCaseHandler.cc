/*
 * SPDX-FileCopyrightText: Copyright 2021-2022 Arm Limited and/or its affiliates
 * <open-source-office@arm.com> SPDX-License-Identifier: Apache-2.0
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

#include "AsrClassifier.hpp"
#include "AsrResult.hpp"
#include "AudioUtils.hpp"
#include "ImageUtils.hpp"
// #include "InputFiles.hpp"
#include "OutputDecode.hpp"
#include "UseCaseCommonUtils.hpp"
#include "Wav2LetterModel.hpp"
#include "Wav2LetterPostprocess.hpp"
#include "Wav2LetterPreprocess.hpp"
#include "hal.h"
#include "log_macros.h"
#include "services_lib_api.h"
#include "services_main.h"

#include <vector>


extern uint32_t m55_comms_handle;
m55_data_payload_t mhu_data;

#define AUDIO_SAMPLES_KWS 32000 
static int16_t audio_inf_kws[AUDIO_SAMPLES_KWS];

namespace arm {
namespace app {

    /**
     * @brief       Presents ASR inference results.
     * @param[in]   results   Vector of ASR classification results to be displayed.
     * @return      true if successful, false otherwise.
     **/
    static bool PresentInferenceResult(const std::vector<asr::AsrResult>& results);


    static void send_name(std::string name)
    {
        
        mhu_data.id = 3; // id for senzmate app

        info("******************* send_name : %s \n", name.c_str());
        strcpy(mhu_data.msg, name.c_str());
        __DMB();
        SERVICES_send_msg(m55_comms_handle, &mhu_data);
           
    }

    /* ASR inference handler. */
    bool ClassifyAudioHandler(ApplicationContext& ctx, uint32_t mode, bool runAll)
    {
        auto& model          = ctx.Get<Model&>("model");
        auto& profiler       = ctx.Get<Profiler&>("profiler");
        auto mfccFrameLen    = ctx.Get<uint32_t>("frameLength");
        auto mfccFrameStride = ctx.Get<uint32_t>("frameStride");
        auto scoreThreshold  = ctx.Get<float>("scoreThreshold");
        auto inputCtxLen     = ctx.Get<uint32_t>("ctxLen");     

        if (!model.IsInited()) {
            printf_err("Model is not initialised! Terminating processing.\n");
            return false;
        }

        TfLiteTensor* inputTensor  = model.GetInputTensor(0);
        TfLiteTensor* outputTensor = model.GetOutputTensor(0);

        /* Get input shape. Dimensions of the tensor should have been verified by
         * the callee. */
        TfLiteIntArray* inputShape = model.GetInputShape(0);

        const uint32_t inputRowsSize = inputShape->data[Wav2LetterModel::ms_inputRowsIdx];
        const uint32_t inputInnerLen = inputRowsSize - (2 * inputCtxLen);
        info(" inputRowsSize : %ld \n", inputRowsSize);
        info(" inputInnerLen : %ld \n", inputInnerLen);

        /* Audio data stride corresponds to inputInnerLen feature vectors. */
        const uint32_t audioDataWindowLen = (inputRowsSize - 1) * mfccFrameStride + (mfccFrameLen);
        const uint32_t audioDataWindowStride = inputInnerLen * mfccFrameStride;

        info(" audioDataWindowLen : %ld \n", audioDataWindowLen);
        info(" audioDataWindowStride : %ld \n", audioDataWindowStride);

        /* NOTE: This is only used for time stamp calculation. */
        const float secondsPerSample = (1.0 / audio::Wav2LetterMFCC::ms_defaultSamplingFreq);
        info(" secondsPerSample : %f \n", secondsPerSample);

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
        // const int16_t* audio_inf = audio_inf_vector.data(); 

        uint32_t audioArrSize = AUDIO_SAMPLES_KWS; // 16000 + 8000;

        static bool audio_inited;

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
                return false;
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
                    return false;
                }

                if (!RunInference(model, profiler)) {
                    printf_err("Inference failed.");
                    return false;
                }

                /* Post processing needs to know if we are on the last audio window. */
                postProcess.m_lastIteration = !audioDataSlider.HasNext();
                if (!postProcess.DoPostProcess()) {
                    printf_err("Post-processing failed.");
                    return false;
                }

                /* Add results from this window to our final results vector. */
                finalResults.emplace_back(asr::AsrResult(
                    singleInfResult,
                    (audioDataSlider.Index() * secondsPerSample * audioDataWindowStride),
                    audioDataSlider.Index(),
                    scoreThreshold));

// #if VERIFY_TEST_OUTPUT
//                 armDumpTensor(outputTensor,
//                               outputTensor->dims->data[Wav2LetterModel::ms_outputColsIdx]);
// #endif        /* VERIFY_TEST_OUTPUT */
            } /* while (audioDataSlider.HasNext()) */

            /* Erase. */
            // str_inf = std::string(str_inf.size(), ' ');
            // hal_lcd_display_text(
            //     str_inf.c_str(), str_inf.size(), dataPsnTxtInfStartX, dataPsnTxtInfStartY, 0);

            ctx.Set<std::vector<asr::AsrResult>>("results", finalResults);

            std::vector<ClassificationResult> combinedResults;
            for (const auto& result : finalResults) {
                combinedResults.insert(
                    combinedResults.end(), result.m_resultVec.begin(), result.m_resultVec.end());
            }

            /* Get the decoded result for the combined result. */
            std::string finalResultStr = audio::asr::DecodeOutput(combinedResults);


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
                    send_name(finalResultStr);
                    break;
                
                default:
                    break;
            }

            // if (!PresentInferenceResult(finalResults)) {
            //     return false;
            // }

            // profiler.PrintProfilingResult();

        } while (runAll); 

        return true;
    }

    static bool PresentInferenceResult(const std::vector<asr::AsrResult>& results)
    {

        /* Results from multiple inferences should be combined before processing. */
        std::vector<ClassificationResult> combinedResults;
        for (const auto& result : results) {
            combinedResults.insert(
                combinedResults.end(), result.m_resultVec.begin(), result.m_resultVec.end());
        }

        /* Get each inference result string using the decoder. */
        for (const auto& result : results) {
            std::string infResultStr = audio::asr::DecodeOutput(result.m_resultVec);
        }

        /* Get the decoded result for the combined result. */
        std::string finalResultStr = audio::asr::DecodeOutput(combinedResults);

        // Check if the result contains "Hi"
        if (finalResultStr.find("Hi") != std::string::npos) {
            info("The word 'Hi' was detected in the recognition result.");
        } else {
            info("The word 'Hi' was not found in the recognition result.");
        }

        info("Complete recognition: %s\n", finalResultStr.c_str());
        send_name(finalResultStr);
        return true;
    }

} /* namespace app */
} /* namespace arm */