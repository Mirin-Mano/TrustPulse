set(${use_case}_ML_FRAMEWORK "TensorFlowLiteMicro")
if (NOT ${use_case}_ML_FRAMEWORK STREQUAL ${ML_FRAMEWORK})
    set(${use_case}_supports_${ML_FRAMEWORK} OFF)
    return()
endif ()

set(${use_case}_supports_${ML_FRAMEWORK} ON)

# Enable OSPI RAM support: frame ring buffers (RhythmFormerProcessing) are placed in
# the .bss.NoInit.activation_buf_dram section (OSPI PSRAM), while the 5 MB
# tensorArena is placed in on-chip SRAM (.bss.NoInit.activation_buf_sram) for
# full-speed NPU execution.
set(OSPI_RAM_SUPPORT ON CACHE BOOL "Enables OSPI RAM driver initialization" FORCE)

# ---------------------------------------------------------------------------
# Enable Alif on-board MIPI/CPI camera backend (not USB UVC).
# The platform HAL selection in hal/source/platform/alif/CMakeLists.txt uses
# ALIF_CAMERA_ENABLED and USB_UVC_ENABLED to pick:
#   ALIF_CAMERA_ENABLED=ON  → camera_alif   (on-board MIPI/CPI camera)
#   USB_UVC_ENABLED=ON      → hal_camera_usb_uvc (USB webcam)
#   (neither)               → hal_camera_static_images (fake/static images)
# Force these values so the use case always gets the correct HAL regardless of
# CMake cache state or board-specific defaults (e.g. DevKit-e1c defaults OFF).
# ---------------------------------------------------------------------------
set(ALIF_CAMERA_ENABLED ON CACHE BOOL
    "Enables the Alif on-board MIPI/CPI camera HAL backend (required for alif_RhythmFormer)" FORCE)

set(USB_UVC_ENABLED OFF CACHE BOOL
    "USB UVC camera disabled — using on-board MIPI/CPI camera for this use case" FORCE)

list(APPEND ${use_case}_API_LIST "rhythmformer" "alif_ui")

USER_OPTION(${use_case}_CAMERA_WIDTH "Camera capture width. HAL hard buffer cap per sensor: MT9M114=320, OV5675=480, ARX3A0=560 — any larger value silently overflows hal_camera_alif.c's static rgb_image buffer producing diagonal stripes/noise. Default 240 fits ALL sensors, is already square (so preprocess center-crop is a no-op), and is rendered 1:1 native pixels on the LCD (no blocky zoom artifacts)."
    240
    STRING)

USER_OPTION(${use_case}_CAMERA_HEIGHT "Camera capture height. Same HAL cap: MT9M114=320, OV5675=480, ARX3A0=560. Default 240 fits ALL sensors."
    240
    STRING)

USER_OPTION(${use_case}_ASSUMED_FPS "Assumed camera frame rate for HR estimation"
    30.0
    STRING)

USER_OPTION(${use_case}_RHYTHMFORMER_SESSION_SECONDS "BVP history buffer length in seconds. Must stay comfortably larger than RHYTHMFORMER_CALIBRATION_SECONDS or the session will roll over before calibration completes."
    30.0
    STRING)

USER_OPTION(${use_case}_RHYTHMFORMER_CALIBRATION_SECONDS "Warm-up/calibration window in seconds before any HR is computed. Raise for a slower but higher-confidence calibration."
    15.0
    STRING)

USER_OPTION(${use_case}_FACE_DETECT_SKIP_FRAMES "Frames to skip after a positive face detection (N skipped -> 1-in-(N+1) detection window)."
    9
    STRING)

USER_OPTION(${use_case}_FACE_DETECT_NO_FACE_THRESHOLD "Consecutive frames without a detected face before recalibration is triggered."
    20
    STRING)

USER_OPTION(${use_case}_FACE_DETECT_MOTION_THRESHOLD "Consecutive frames of detected motion before recalibration is triggered. Raise if single-frame motion (blinks/micro-movements) false-triggers recalibration too often."
    5
    STRING)

USER_OPTION(${use_case}_ACTIVATION_BUF_SZ "Activation buffer size for the RhythmFormer model"
    0x00500000
    STRING)

USER_OPTION(${use_case}_SHOW_INF_TIME "Show inference time"
    ON
    BOOL)

USER_OPTION(${use_case}_SHOW_PIPELINE_TIME "Show pipeline time"
    ON
    BOOL)

set(RHYTHMFORMER_RESOURCES_DIR ${RESOURCES_PATH}/rhythmformer)

if (ETHOS_U_NPU_ENABLED)
    set(DEFAULT_MODEL_PATH
        ${RHYTHMFORMER_RESOURCES_DIR}/RhythmFormer_fd10_72x72_efficient_int8_vela_${ETHOS_U_NPU_CONFIG_ID}.tflite)
else()
    set(DEFAULT_MODEL_PATH
        ${RHYTHMFORMER_RESOURCES_DIR}/RhythmFormer_fd10_72x72_efficient_int8.tflite)
endif()

USER_OPTION(${use_case}_MODEL_PATH "RhythmFormer tflite model file"
    ${DEFAULT_MODEL_PATH}
    FILEPATH)

generate_model_code(
    MODEL_PATH ${${use_case}_MODEL_PATH}
    DESTINATION ${SRC_GEN_DIR}
    NAMESPACE   "arm" "app" "rhythmformer")

# ---------------------------------------------------------------------------
# FOMO face-detection model (72x72 int8, Vela-compiled)
# Shared arena: both models initialise against the same tensorArena; they
# are never run simultaneously, so the arena size just needs to cover the
# larger of the two model scratch requirements (RhythmFormer at 5 MB already covers
# the much smaller FOMO model).
# ---------------------------------------------------------------------------
set(DEFAULT_FOMO_MODEL_PATH
    ${RHYTHMFORMER_RESOURCES_DIR}/fomo-face-detection-72x72-int8_vela.tflite)

USER_OPTION(${use_case}_FOMO_MODEL_PATH
    "FOMO 72x72 int8 face-detection model (Vela-compiled .tflite)"
    ${DEFAULT_FOMO_MODEL_PATH}
    FILEPATH)

if (NOT EXISTS "${${use_case}_FOMO_MODEL_PATH}")
    message(WARNING
        "alif_RhythmFormer: FOMO model not found at "
        "'${${use_case}_FOMO_MODEL_PATH}'. "
        "Ensure fomo-face-detection-72x72-int8_vela.tflite is at the repo root "
        "(or set -Dalif_RhythmFormer_FOMO_MODEL_PATH=<path>) and reconfigure.")
endif()

generate_model_code(
    MODEL_PATH  ${${use_case}_FOMO_MODEL_PATH}
    DESTINATION ${SRC_GEN_DIR}
    NAMESPACE   "arm" "app" "fomofacedet")

USER_OPTION(${use_case}_FACE_DETECT_CONF_THRESHOLD
    "FOMO face-detection confidence threshold (0.0-1.0), default 0.8. Raise to reduce false positives from profile/partial faces."
    0.8
    STRING)

# ---------------------------------------------------------------------------
# MotionDetector tuning knobs (standalone luma-diff motion gate — see
# MotionDetector.hpp for full rationale). Defaults mirror the values that
# used to live inside HybridFaceDetector's Stage 1.
# ---------------------------------------------------------------------------
USER_OPTION(${use_case}_MOTION_DETECT_W "MotionDetector downsampled grid width (pixels)."
    64
    STRING)

USER_OPTION(${use_case}_MOTION_DETECT_H "MotionDetector downsampled grid height (pixels)."
    64
    STRING)

USER_OPTION(${use_case}_MOTION_DETECT_MIN_PCT "Minimum percent of the downsampled grid that must differ by >= MOTION_DETECT_DIFF luma counts before a frame counts as motion. Set to 0 to disable the motion gate entirely."
    1
    STRING)

USER_OPTION(${use_case}_MOTION_DETECT_DIFF "Luma delta (0-255) for a pixel to count as moved in the MotionDetector grid diff."
    15
    STRING)

USER_OPTION(${use_case}_MOTION_DETECT_BYPASS_ON_FIRST "If ON, MotionDetector::DetectMotion()'s first call after construction/reset (no previous frame to diff against) reports 'no motion' instead of a false-positive trigger."
    ON
    BOOL)

USER_OPTION(${use_case}_ENABLE_MOTION_DETECTOR "Enable the CPU luma-diff MotionDetector stage. If OFF, DetectMotion() is never called (saving the downsample+diff CPU cost each detection frame) and motion is always treated as absent, so recalibration can only be triggered by FACE_DETECT_NO_FACE_THRESHOLD (face loss)."
    ON
    BOOL)

set(${use_case}_COMPILE_DEFS
    "CAMERA_WIDTH=${${use_case}_CAMERA_WIDTH}"
    "CAMERA_HEIGHT=${${use_case}_CAMERA_HEIGHT}"
    "RHYTHMFORMER_ASSUMED_FPS=${${use_case}_ASSUMED_FPS}f"
    "SHOW_INF_TIME=$<BOOL:${${use_case}_SHOW_INF_TIME}>"
    "SHOW_PIPELINE_TIME=$<BOOL:${${use_case}_SHOW_PIPELINE_TIME}>"
    "FACE_DETECT_CONF_THRESHOLD=${${use_case}_FACE_DETECT_CONF_THRESHOLD}f"
    "RHYTHMFORMER_SESSION_SECONDS=${${use_case}_RHYTHMFORMER_SESSION_SECONDS}f"
    "RHYTHMFORMER_CALIBRATION_SECONDS=${${use_case}_RHYTHMFORMER_CALIBRATION_SECONDS}f"
    "FACE_DETECT_SKIP_FRAMES=${${use_case}_FACE_DETECT_SKIP_FRAMES}"
    "FACE_DETECT_NO_FACE_THRESHOLD=${${use_case}_FACE_DETECT_NO_FACE_THRESHOLD}"
    "FACE_DETECT_MOTION_THRESHOLD=${${use_case}_FACE_DETECT_MOTION_THRESHOLD}"
    "MOTION_DETECT_W=${${use_case}_MOTION_DETECT_W}"
    "MOTION_DETECT_H=${${use_case}_MOTION_DETECT_H}"
    "MOTION_DETECT_MIN_PCT=${${use_case}_MOTION_DETECT_MIN_PCT}"
    "MOTION_DETECT_DIFF=${${use_case}_MOTION_DETECT_DIFF}"
    "MOTION_DETECT_BYPASS_ON_FIRST=$<BOOL:${${use_case}_MOTION_DETECT_BYPASS_ON_FIRST}>"
    "ENABLE_MOTION_DETECTOR=$<BOOL:${${use_case}_ENABLE_MOTION_DETECTOR}>")