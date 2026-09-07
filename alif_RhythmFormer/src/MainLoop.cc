#include "hal.h"
#include "mlek/fwk/tflm/RhythmFormerModel.hpp"
#include "mlek/fwk/tflm/FomoFaceModel.hpp"
#include "UseCaseHandler.hpp"
#include "UseCaseCommonUtils.hpp"
#include "mlek/log/log_macros.h"
#include "BufAttributes.hpp"
#include "mlek/use_case/rhythmformer/RhythmFormerProcessing.hpp"

namespace arm {
namespace app {

#ifdef ACTIVATION_BUF_ATTRIBUTE
#undef ACTIVATION_BUF_ATTRIBUTE
#endif

#define RHYTHMFORMER_ACTIVATION_BUF_SECTION section(".bss.NoInit.activation_buf_sram")
#define ACTIVATION_BUF_ATTRIBUTE  __attribute__((aligned(16), RHYTHMFORMER_ACTIVATION_BUF_SECTION))

#define FOMO_ACTIVATION_BUF_SECTION section(".bss.NoInit.activation_buf_dram")
#define FOMO_ACTIVATION_BUF_ATTRIBUTE __attribute__((aligned(16), FOMO_ACTIVATION_BUF_SECTION))

    static constexpr size_t kFomoActivationBufSz = 0x00040000; /* 256 KiB — generous for 72x72 FOMO model */

    static uint8_t tensorArena[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;
    static uint8_t fomoTensorArena[kFomoActivationBufSz] FOMO_ACTIVATION_BUF_ATTRIBUTE;

    namespace rhythmformer {
        extern uint8_t* GetModelPointer();
        extern size_t GetModelLen();
    } /* namespace rhythmformer */

    namespace fomofacedet {
        extern uint8_t* GetModelPointer();
        extern size_t GetModelLen();
    } /* namespace fomofacedet */

} /* namespace app */
} /* namespace arm */

#ifndef RHYTHMFORMER_ASSUMED_FPS
#define RHYTHMFORMER_ASSUMED_FPS 30.0f
#endif

#ifndef CAMERA_WIDTH
#define CAMERA_WIDTH 240
#endif

#ifndef CAMERA_HEIGHT
#define CAMERA_HEIGHT 240
#endif

void MainLoop()
{
    /* -----------------------------------------------------------------------
     * 1. RhythmFormer model — initialise first; creates the MicroAllocator from the
     *    shared tensorArena.  Init() dumps interpreter/tensor info to UART.
     * ----------------------------------------------------------------------- */
    arm::app::fwk::tflm::RhythmFormerModel model;

    arm::app::fwk::iface::MemoryRegion modelMem{arm::app::rhythmformer::GetModelPointer(),
                                                arm::app::rhythmformer::GetModelLen()};
    arm::app::fwk::iface::MemoryRegion computeMem{arm::app::tensorArena,
                                                  sizeof(arm::app::tensorArena)};

    info("===== RhythmFormer model init (blob %zu bytes) =====\n",
         arm::app::rhythmformer::GetModelLen());
    if (!model.Init(computeMem, modelMem)) {
        printf_err("Failed to initialise RhythmFormer model\n");
        return;
    }

    /* -----------------------------------------------------------------------
     * 2. FOMO face detection model — uses its own separate tensor arena in
     *    OSPI PSRAM (.bss.NoInit.activation_buf_dram), not shared with the
     *    RhythmFormer model's MicroAllocator: TFLM's MicroAllocator is
     *    stateful and only supports one AllocateTensors() pass, so sharing
     *    it between two interpreters desyncs tensor pointers from what
     *    Invoke() actually writes. FOMO's 72×72 model has a tiny activation
     *    footprint and only runs ~1 in 10 frames, so the extra PSRAM arena
     *    costs little and face-detection latency isn't on the rPPG
     *    inference critical path.
     * ----------------------------------------------------------------------- */
    arm::app::fwk::tflm::FomoFaceModel faceModel;

    arm::app::fwk::iface::MemoryRegion faceModelMem{arm::app::fomofacedet::GetModelPointer(),
                                                    arm::app::fomofacedet::GetModelLen()};
    arm::app::fwk::iface::MemoryRegion ufComputeMem{arm::app::fomoTensorArena,
                                                    sizeof(arm::app::fomoTensorArena)};

    info("===== FOMO Face model init (blob %zu bytes, arena %zu bytes @ 0x%p) =====\n",
         arm::app::fomofacedet::GetModelLen(),
         sizeof(arm::app::fomoTensorArena),
         static_cast<void*>(arm::app::fomoTensorArena));

    if (!faceModel.Init(ufComputeMem, faceModelMem)) {
        printf_err("Failed to initialise FOMO face-detection model\n");
        return;
    }

    info("===== FOMO Face interpreter (post-init) =====\n");
    faceModel.LogInterpreterInfo();

    /* -----------------------------------------------------------------------
     * 3. Use-case handler init (camera, LVGL).
     * ----------------------------------------------------------------------- */
    if (!alif::app::RhythmFormerHandlerInit(model)) {
        printf_err("Failed to initialise RhythmFormer use case handler\n");
        return;
    }

    info("RhythmFormer model loaded. Input shape: [%d, %d, %d, %d]\n",
         model.GetInputTensor(0)->Shape()[arm::app::fwk::tflm::RhythmFormerModel::ms_frameDepthIdx],
         model.GetInputTensor(0)->Shape()[arm::app::fwk::tflm::RhythmFormerModel::ms_inputRowsIdx],
         model.GetInputTensor(0)->Shape()[arm::app::fwk::tflm::RhythmFormerModel::ms_inputColsIdx],
         model.GetInputTensor(0)->Shape()[arm::app::fwk::tflm::RhythmFormerModel::ms_inputChannelsIdx]);

    info("FOMO model loaded. Input shape: [%d, %d, %d, %d]\n",
         faceModel.GetInputTensor(0)->Shape()[arm::app::fwk::tflm::FomoFaceModel::ms_inputBatchIdx],
         faceModel.GetInputTensor(0)->Shape()[arm::app::fwk::tflm::FomoFaceModel::ms_inputRowsIdx],
         faceModel.GetInputTensor(0)->Shape()[arm::app::fwk::tflm::FomoFaceModel::ms_inputColsIdx],
         faceModel.GetInputTensor(0)->Shape()[arm::app::fwk::tflm::FomoFaceModel::ms_inputChannelsIdx]);

    /* -----------------------------------------------------------------------
     * 4. Build application context.
     * ----------------------------------------------------------------------- */
    arm::app::ApplicationContext caseContext;

    arm::app::Profiler profiler{"rhythmformer"};
    caseContext.Set<arm::app::Profiler&>("profiler", profiler);
    caseContext.Set<arm::app::fwk::iface::Model&>("model", model);
    caseContext.Set<arm::app::fwk::iface::Model&>("faceModel", faceModel);

    static arm::app::RhythmFormerResult result{};
    caseContext.Set<arm::app::RhythmFormerResult&>("result", result);

    static arm::app::RhythmFormerPreProcess preProcess(model.GetInputTensor(0), CAMERA_WIDTH, CAMERA_HEIGHT);
    caseContext.Set<arm::app::RhythmFormerPreProcess&>("preProcess", preProcess);

    static arm::app::RhythmFormerPostProcess postProcess(
        model.GetOutputTensor(0), result, RHYTHMFORMER_ASSUMED_FPS);
    caseContext.Set<arm::app::RhythmFormerPostProcess&>("postProcess", postProcess);

    static arm::app::FaceDetectionState faceState{};
    caseContext.Set<arm::app::FaceDetectionState&>("faceState", faceState);

    info("Starting RhythmFormer rPPG inference loop (assumed fps=%.1f)...\n",
         static_cast<double>(RHYTHMFORMER_ASSUMED_FPS));

    do {
        alif::app::RhythmFormerHandler(caseContext);
    } while (1);
}
