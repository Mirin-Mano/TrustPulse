# Alif RhythmFormer – Build and Flash Guide

This guide explains how to build and run the `alif_RhythmFormer` use case using the [Alif ML Embedded Evaluation Kit](https://github.com/alifsemi/alif_ml-embedded-evaluation-kit).

## About This Application

This application implements a live remote heart-rate estimation pipeline that uses the on-board camera for face detection and rPPG-based heart-rate tracking.

Key features:
- Live camera capture pipeline
- Face detection using a FOMO model
- Heart-rate estimation using the RhythmFormer (TSCAN) model
- Optimized for Alif Semiconductor devices

## Prerequisites

Before proceeding, follow the complete setup instructions from:

- [ML_Embedded_Evaluation_Kit.md](https://github.com/alifsemi/alif_ml-embedded-evaluation-kit/blob/main/ML_Embedded_Evaluation_Kit.md)

This includes:

- Toolchain installation
- Environment setup
- CMSIS and Ethos-U dependencies
- Board setup

---

# Add the RhythmFormer Use Case Files


```bash
cd alif_ml-embedded-evaluation-kit

# Application sources
cp -r <trustpulse>/alif_RhythmFormer source/app/use_case/alif_RhythmFormer
rm -f source/app/use_case/alif_RhythmFormer/CMakeLists.txt

# Shared RhythmFormer library
cp -r <trustpulse>/rhythmformer source/lib/mlek/use_case/rhythmformer

# TFLM model wrappers (merge into the existing tflm folder)
cp <trustpulse>/tflm/RhythmFormerModel.hpp <trustpulse>/tflm/RhythmFormerModel.cc \
   <trustpulse>/tflm/FomoFaceModel.hpp   <trustpulse>/tflm/FomoFaceModel.cc \
   source/lib/mlek/fwk/tflm/
```


- add `rhythmformer` to `_MLEK_API_LIST` in `source/lib/mlek/use_case/CMakeLists.txt`
- add `RhythmFormerModel.cc` and `FomoFaceModel.cc` to
  `source/lib/mlek/fwk/tflm/CMakeLists.txt`


---

# Prepare the Model Files

RhythmFormer needs **two** models in place: the TSCAN heart-rate model and the
FOMO face-detection model.

### On Linux/macOS:

First, ensure the target directory exists:

```bash
mkdir -p ${MLEK_ROOT}/resources/rhythmformer
```

Copy the TSCAN model (use the Vela-compiled variant for an Ethos-U build):

```bash
cp <trustpulse>/TSCAN_fd10_72x72_efficient_int8_vela_Z256.tflite \
   ${MLEK_ROOT}/resources/rhythmformer/
```

Copy the FOMO face-detection model to:

```bash
cp <trustpulse>/fomo-face-detection-72x72-int8_vela.tflite \
   ${MLEK_ROOT}/resources/rhythmformer/
```

---

# Build the Application

Return to the root directory:

```bash
alif_ml-embedded-evaluation-kit
```

Configure the build:

```bash
cmake  -B build_alif_RhythmFormer -DTARGET_PLATFORM=alif \
 -DUSE_CASE_BUILD=alif_RhythmFormer \
 -DTARGET_SUBSYSTEM=RTSS-HP \
 -DTARGET_BOARD=DevKit-e8 \
 -DCMAKE_TOOLCHAIN_FILE=../scripts/cmake/toolchains/bare-metal-gcc.cmake \
 -DGLCD_UI=ON \
 -DLINKER_SCRIPT_NAME=RTSS-HP \
 -DCMAKE_BUILD_TYPE=Release -DMLEK_LOG_LEVEL=MLEK_LOG_LEVEL_INFO \
 -DETHOS_U_NPU_ID=U85 \
 -DCONSOLE_UART=4 \
 -DOSPI_RAM_SUPPORT=ON \
 -DUSB_UVC_ENABLED=OFF \
 -DALIF_CAMERA_ENABLED=ON \
 -Dalif_rhythmformer_CAMERA_WIDTH=240 \
 -Dalif_rhythmformer_CAMERA_HEIGHT=240 \
 -Dalif_rhythmformer_FOMO_MODEL_PATH=./fomo-face-detection-72x72-int8_vela.tflite \
 -Dalif_rhythmformer_FACE_DETECT_CONF_THRESHOLD=0.04 \
 -Dalif_rhythmformer_ENABLE_MOTION_DETECTOR=OFF \
 -Dalif_rhythmformer_RHYTHMFORMER_CALIBRATION_SECONDS=5.0 \
 -Dalif_rhythmformer_RHYTHMFORMER_SESSION_SECONDS=5.0 ..
```


During configure, confirm CMake reports:

```text
Found sources for use-case alif_RhythmFormer
Building use-cases: alif_RhythmFormer
Using rhythmformer_api for alif_RhythmFormer
```

Build the project:

```bash
cd build_alif_RhythmFormer
```

```bash
make -j4
```

> Adjust `-j4` according to the number of CPU cores available on your system.

If CMake keeps using an old model path or camera size, delete the build
directory (or at least `CMakeCache.txt`) and configure again.

---

# Generated Output Files

After a successful build, navigate to:

```bash
cd bin/sectors/alif_RhythmFormer
```

You should find:

- `mram.bin` – Only this file is needed!

---

# Flash the Application



- Flash `mram.bin` into MRAM using SE Tools

---

# Run the Application

On boot:

- The application initializes the on-board camera
- Starts live capture pipeline
- Loads the FOMO face-detection model and locates a face
- Loads the RhythmFormer  model
- Runs real-time rPPG-based heart-rate estimation
- Displays camera feed and heart-rate result on the display, once past the
  configured calibration period
