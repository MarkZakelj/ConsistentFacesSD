import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort
import scipy
from loguru import logger
from scipy.stats import entropy
from tqdm import tqdm

from utils.imgs import (
    get_face_imgs,
    get_identity_img,
    get_img_info,
    get_number_of_images,
    get_prompt_seed_pairs,
    get_subset_info,
    update_img_info,
)
from utils.paths import MODELS_DIR

skip_subsets = ["identities"]

RESNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
RESNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

# def mapping_compositum(mapping1: list, mapping2: list):
#     """
#     Combine two mappings into one
#     :param mapping1: list of mappings [(idx1, idx2) ...]
#     :param mapping2: list of mappings [(idx1, idx2) ...]
#     :return: list of mappings [(idx1, idx2) ...]
#     """
#     mapping = {}
#     for idx1, idx2 in mapping1:
#         if idx2 == -1:
#             continue
#         mapping[idx1] = idx2
#     for idx1, idx2 in mapping2:
#         if idx2 == -1:
#             continue
#         mapping[idx1] = idx2
#     return list(mapping.items())


def preprocess_image(image_batch: list, resize=None):
    """
    turn list of numpy images into a preprocessed numpy batch
    list of RGB images in the format [H, W, C], numpy ndarray, uint8"""
    if len(image_batch) == 0:
        return []
    if resize is None:
        resize = image_batch[0].shape[1:3]
    w, h = resize
    image_batch = np.array(
        [cv2.resize(image, (w, h)).astype(np.float32) / 255.0 for image in image_batch]
    )
    image_batch = image_batch.transpose((0, 3, 1, 2))  # BCHW format
    b = image_batch.shape[0]
    # Apply normalization for resnet
    image_batch = (image_batch - np.repeat(RESNET_MEAN, b, axis=0)) / np.repeat(
        RESNET_STD, b, axis=0
    )
    return image_batch


def js_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    p /= p.sum()
    q /= q.sum()

    # Calculate the M distribution
    m = 0.5 * (p + q)

    # Calculate the Jensen-Shannon Divergence
    js_div = 0.5 * (entropy(p, m) + entropy(q, m))
    return js_div


def softmax(x):
    # Subtract the max for numerical stability
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    # Divide by the sum of exps
    return e_x / np.sum(e_x, axis=1, keepdims=True)


label_map = {"sex": {0: "female", 1: "male"}, "age": {0: "adult", 1: "child"}}


class AgeSexInference:
    def __init__(self, onnx_model_path):
        self.model = ort.InferenceSession(onnx_model_path)

    def predict(self, image: np.ndarray):
        """RGB image in the format [B, C, H, W], numpy ndarray"""
        age, sex = self.model.run(None, {"face_image": image})
        return age, sex

    def predict_probs(self, image: np.ndarray):
        """RGB image in the format [B, C, H, W]"""
        age, sex = self.predict(image)
        return softmax(age), softmax(sex)

    def predict_labels(self, image: np.ndarray):
        """RGB image in the format [B, C, H, W]"""
        age, sex = self.predict(image)
        age_idx = np.argmax(age, axis=1)
        sex_idx = np.argmax(sex, axis=1)
        return (
            [label_map["age"][idx] for idx in age_idx],
            [label_map["sex"][idx] for idx in sex_idx],
        )


age_groups = [(8, 0.01), (16, 0.4), (200000000000, 0.99)]


def select_age_group(ages):
    results = []
    for age in ages:
        for age_max, prob in age_groups:
            if age < age_max:
                results.append([prob, 1 - prob])
                break
    return np.array(results).astype(np.float32)


class AgeSexInference2:
    def __init__(self, onnx_model_path):
        self.model = ort.InferenceSession(onnx_model_path)

    def predict(self, image: np.ndarray):
        """RGB image in the format [B, C, H, W], numpy ndarray"""
        out = self.model.run(None, {"data": image})[0]
        # age in the output is float
        age = select_age_group(out[:, 2] * 100)
        sex = softmax(out[:, :2])
        return age, sex

    def predict_probs(self, image: np.ndarray):
        """RGB image in the format [B, C, H, W]"""
        return self.predict(image)


class FaceMatcher:
    def __init__(self):
        onnx_model_path = os.path.join(MODELS_DIR, "model_inception_resnet.onnx")
        self.model = AgeSexInference(onnx_model_path)

    def match_faces(
        self, input_faces: list[np.ndarray], target_faces: list[np.ndarray]
    ):
        if len(input_faces) == 0 or len(target_faces) == 0:
            return None
        input_faces = preprocess_image(input_faces, resize=(160, 160))
        target_faces = preprocess_image(target_faces, resize=(160, 160))
        age_i, sex_i = self.model.predict_probs(input_faces)
        age_t, sex_t = self.model.predict_probs(target_faces)
        distances = np.zeros((len(input_faces), len(target_faces)))
        for i in range(len(input_faces)):
            for j in range(len(target_faces)):
                distances[i, j] = 0.7 * js_divergence(
                    age_i[i], age_t[j]
                ) + 0.3 * js_divergence(sex_i[i], sex_t[j])
        rows, cols = scipy.optimize.linear_sum_assignment(distances)
        mapping = list(zip(rows, cols))
        mapping = sorted(mapping, key=lambda x: x[0])  # not sure if needed
        full_mapping = [-1] * len(input_faces)
        for a, b in mapping:
            full_mapping[a] = b
        # print("Face Mapping: ", full_mapping)
        return full_mapping


def calculate_iou(bbox1: list, bbox2: list) -> float:
    """
    calculate the intersection over union between two bboxes
    :param bbox1: bbox in the format [x1, y1, x2, y2]
    :param bbox2: bbox in the format [x1, y1, x2, y2]
    :return: IoU
    """
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    x5, y5 = max(x1, x3), max(y1, y3)
    x6, y6 = min(x2, x4), min(y2, y4)
    if x5 >= x6 or y5 >= y6:
        return 0
    intersection = (x6 - x5) * (y6 - y5)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union = area1 + area2 - intersection
    return intersection / union


def bbox_mapping(bboxes1: list, bboxes2: list) -> dict:
    """
    map bboxes1 to bboxes2 considering the IoU between them
    :param bboxes1: list of bbox in the format [x1, y1, x2, y2]
    :param bboxes2: list of bbox in the format [x1, y1, x2, y2]
    :return: list of mapping [(idx1, idx2) ...]
    """
    mapping = {}
    for i, bbox1 in enumerate(bboxes1):
        max_iou = 0
        max_idx = -1
        for j, bbox2 in enumerate(bboxes2):
            iou = calculate_iou(bbox1, bbox2)
            if iou > max_iou:
                max_iou = iou
                max_idx = j
        mapping[i] = max_idx
    return mapping


def extract_all_identities_with_facematch_on_subset(subset_name: str):
    if subset_name in skip_subsets:
        return
    logger.info("SUBSET NAME: " + subset_name)

    subset_info = get_subset_info(subset_name)
    n_people = len(subset_info["person_codes"])
    base_subset_name = "base_one_person_dreamshaper"
    if n_people == 2:
        base_subset_name = "base_two_people_dreamshaper"
    if n_people == 3:
        base_subset_name = "base_three_people_dreamshaper"
    if n_people > 3:
        raise ValueError("Only 1, 2, or 3 people are supported.")
    prompt_seed_pairs = get_prompt_seed_pairs(n_people)
    n_images = get_number_of_images(subset_name)
    matcher = FaceMatcher()
    for i in tqdm(range(n_images)):
        info = get_img_info(subset_name, i)
        base_info = get_img_info(base_subset_name, i)
        if "face_info" not in info:
            logger.warning(
                f"No face info for image {i} - do you need to run face detection?"
            )
            continue
        do = True
        for ff in info["face_info"]:
            if "identity" in ff:
                do = False
                break
        if not do:
            continue
        person_codes = list(prompt_seed_pairs[i]["people"].values())
        input_faces = [get_identity_img(code) for code in person_codes]
        target_faces = get_face_imgs(base_subset_name, i)
        mapping = matcher.match_faces(input_faces, target_faces)
        if mapping is None:
            continue
        bbox_base = [
            base_info["face_info"][idx]["bbox"]
            for idx in range(len(base_info["face_info"]))
        ]
        bbox_new = [
            info["face_info"][idx]["bbox"] for idx in range(len(info["face_info"]))
        ]
        bbox_map = bbox_mapping(bbox_base, bbox_new)
        for k, idx in enumerate(mapping):
            if idx == -1:
                continue
            info["face_info"][bbox_map[idx]]["identity"] = person_codes[k]
        update_img_info(subset_name, i, info)


def main():
    parser = argparse.ArgumentParser(
        description="Face-matching identity extraction for a given subset of images."
    )
    parser.add_argument(
        "--subset-name",
        type=str,
        help="The name of the subset to process images for.",
        required=False,
    )
    args = parser.parse_args()
    subset_name = args.subset_name
    subsets = ["two_people_faceid_dreamshaper"]
    if subset_name is not None:
        subsets = [subset_name]
    for subset in subsets:
        extract_all_identities_with_facematch_on_subset(subset)


if __name__ == "__main__":
    main()
