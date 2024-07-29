import json

import torch.backends.mps

from pose_extraction.dwpose import DwposeDetector

bbox_detector = [
    "yolox_l.torchscript.pt",
    "yolox_l.onnx",
    "yolo_nas_l_fp16.onnx",
    "yolo_nas_m_fp16.onnx",
    "yolo_nas_s_fp16.onnx",
]
pose_estimator = [
    "dw-ll_ucoco_384_bs5.torchscript.pt",
    "dw-ll_ucoco_384.onnx",
    "dw-ll_ucoco.onnx",
]


DWPOSE_MODEL_NAME = "yzd-v/DWPose"


def get_torch_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_dw_pose_model():
    return DwposeDetector.from_pretrained(
        DWPOSE_MODEL_NAME,
        DWPOSE_MODEL_NAME,
        "yolox_l.onnx",
        "dw-ll_ucoco_384.onnx",
    )
