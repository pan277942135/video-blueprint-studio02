#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PIP=("${PYTHON_BIN}" -m pip)

"${PIP[@]}" install --upgrade pip

# The certified OpenMMLab/Torch CPU stack uses extensions built against NumPy 1.x.
# Keep one cv2 provider: opencv-contrib-python also satisfies MediaPipe's cv2 needs.
"${PIP[@]}" install \
  "numpy==1.26.4" \
  "opencv-contrib-python==4.10.0.84"

"${PIP[@]}" install \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.1.0" \
  "torchvision==0.16.0"

"${PIP[@]}" install "mmengine>=0.7.1,<1.0.0"
"${PIP[@]}" install \
  "mmcv==2.1.0" \
  -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html
"${PIP[@]}" install "mmdet==3.3.0"
"${PIP[@]}" install "mmpose==1.3.2" --no-deps
"${PIP[@]}" install \
  json_tricks \
  matplotlib \
  munkres \
  pillow \
  scipy \
  "xtcocotools>=1.12"

# mediapipe==0.10.35 currently resolves an unconstrained OpenCV 5 / NumPy 2 pair.
# That breaks the certified Torch/MMCV NumPy-1.x ABI. Install the approved MediaPipe
# wheel without dependency resolution, then add only the non-cv2 dependencies needed
# by its Python runtime while retaining the certified NumPy/OpenCV pair above.
"${PIP[@]}" install "mediapipe==0.10.35" --no-deps
"${PIP[@]}" install \
  "absl-py~=2.3" \
  "flatbuffers~=25.9" \
  "sounddevice~=0.5"

"${PIP[@]}" install \
  "pydantic>=2.0.0" \
  "jsonschema>=4.18.0" \
  "scenedetect>=0.6.7,<0.7"

# Install this repository without allowing its generic dependency set to replace
# the certified E12 runtime versions above.
"${PIP[@]}" install -e . --no-deps

"${PYTHON_BIN}" - <<'PY'
import cv2
import mediapipe as mp
import mmdet
import mmpose
import numpy as np
import torch

versions = {
    "mediapipe": mp.__version__,
    "mmdet": mmdet.__version__,
    "mmpose": mmpose.__version__,
    "numpy": np.__version__,
    "opencv": cv2.__version__,
    "torch": torch.__version__,
}
for name, value in versions.items():
    print(f"{name} {value}")

assert versions["mediapipe"] == "0.10.35", versions
assert versions["mmdet"] == "3.3.0", versions
assert versions["mmpose"] == "1.3.2", versions
assert versions["numpy"] == "1.26.4", versions
assert versions["opencv"].startswith("4.10.0"), versions
assert versions["torch"].startswith("2.1.0"), versions
PY
