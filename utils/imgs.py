import base64
import json
import os
from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image

from utils.paths import DATA_DIR, OUTPUT_DIR


def prepare_img_code(img_code: int | str):
    if isinstance(img_code, str):
        return img_code
    return f"{img_code:08d}"


def img_2_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def base64_2_img(base64_string: str) -> Image.Image:
    binary_data = base64.b64decode(base64_string)
    image_bytes = BytesIO(binary_data)
    pillow_image = Image.open(image_bytes)
    return pillow_image


def read_base64_image(path: str, as_image=False) -> str | Image.Image:
    with open(path) as f:
        base64_string = f.read().strip()
    if as_image:
        return base64_2_img(base64_string)
    return base64_string


def np_2_tensor(img: np.ndarray):
    """
    Convert a numpy array to a PyTorch tensor uint8
    :param img: np.ndarray - image to convert, (H, W, C)
    :return: tensor: torch.Tensor - converted image (B, C, H, W)
    """
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


def get_img_paths_in_dir(directory):
    types = (".jpg", ".png", ".jpeg")  # the tuple of file types
    img_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(types):
                img_paths.append(os.path.join(root, file))
    return img_paths


def count_images_in_dir(directory):
    types = (".jpg", ".png", ".jpeg")  # the tuple of file types
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(types):
                count += 1
    return count


def get_subset_info(subset_name: str):
    sub_dir = os.path.join(OUTPUT_DIR, subset_name)
    info = json.load(open(os.path.join(sub_dir, "info.json")))
    return info


def get_prompt_seed_pairs(n_people):
    return json.load(
        open(os.path.join(DATA_DIR, f"prompt_seed_pairs_{n_people}.json"), "r")
    )


def get_img_info(subset_name: str, img_code: int | str):
    sub_dir = os.path.join(OUTPUT_DIR, subset_name)
    info = json.load(
        open(os.path.join(sub_dir, "img_info", prepare_img_code(img_code) + ".json"))
    )
    return info


def get_img(subset_name: str, img_code: int | str):
    """
    Get the image from the subset. numpy format, HWC-RGB-uint8
    :param subset_name:
    :param img_code:
    :return:
    """
    sub_dir = os.path.join(OUTPUT_DIR, subset_name)
    img_path = os.path.join(sub_dir, "images", prepare_img_code(img_code) + ".jpg")
    img = cv2.imread(img_path)[..., [2, 1, 0]]  # BGR to RGB
    return img


def get_img_codes(subset_name: str):
    sub_dir = os.path.join(OUTPUT_DIR, subset_name)
    img_codes = []
    for root, dirs, files in os.walk(os.path.join(sub_dir, "images")):
        for file in files:
            if file.endswith(".jpg"):
                img_code = file.rsplit(".", maxsplit=1)[0]
                img_codes.append(img_code)
    return img_codes


def get_face_imgs(subset_name: str, img_code: int | str, image_pad=0.2):
    img, info = get_image_and_info(subset_name, img_code)
    face_imgs = []
    if "face_info" not in info:
        return []
    face_info = info["face_info"]
    img_shape = img.shape[:2]
    for face in face_info:
        bbox = face["bbox"]
        bbox = pad_bbox(bbox, img_shape, image_pad)
        bbox = square_bbox(bbox, img_shape)
        if not is_valid_bbox(bbox, img_shape):
            raise ValueError("Invalid bounding box")
        x1, y1, x2, y2 = bbox
        face_img = img[y1:y2, x1:x2]
        face_imgs.append(face_img)
    del img
    del info
    return face_imgs


def get_img2(subset_name: str, img_code: int | str):
    sub_dir = os.path.join(OUTPUT_DIR, subset_name)
    img_path = os.path.join(sub_dir, "images", prepare_img_code(img_code) + ".jpg")
    img = Image.open(img_path)
    return np.ascontiguousarray(img, dtype=np.uint8)


def get_image_and_info(subset_name: str, img_code: int | str):
    img = get_img(subset_name, img_code)
    info = get_img_info(subset_name, img_code)
    return img, info


def update_img_info(
    subset_name: str, img_code: int | str, info_update: dict, replace=False
):
    info = get_img_info(subset_name, img_code)
    info = info | info_update
    if replace:
        info = info_update
    sub_dir = os.path.join(OUTPUT_DIR, subset_name)
    info_path = os.path.join(sub_dir, "img_info", prepare_img_code(img_code) + ".json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)


def get_number_of_images(subset_name: str):
    if not subset_exists(subset_name):
        raise ValueError(f"Subset {subset_name} does not exist")
    sub_dir = os.path.join(OUTPUT_DIR, subset_name)
    return count_images_in_dir(os.path.join(sub_dir, "images"))


def subset_exists(subset_name: str):
    return os.path.exists(os.path.join(OUTPUT_DIR, subset_name))


def get_image_paths(subset_name: str, img_folder: str = "images"):
    sub_dir = os.path.join(OUTPUT_DIR, subset_name)
    return get_img_paths_in_dir(os.path.join(sub_dir, img_folder))


def get_info_paths(subset_name: str, info_folder: str = "img_info"):
    directory = os.path.join(OUTPUT_DIR, subset_name, info_folder)
    info_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                info_paths.append(os.path.join(root, file))
    return info_paths


def get_full_info_path(subset_name: str, img_code: int | str):
    sub_dir = os.path.join(OUTPUT_DIR, subset_name)
    return os.path.join(sub_dir, "img_info", prepare_img_code(img_code) + ".json")


def get_full_img_path(subset_name: str, img_code: int | str):
    sub_dir = os.path.join(OUTPUT_DIR, subset_name)
    return os.path.join(sub_dir, "images", prepare_img_code(img_code) + ".jpg")


def image_exists(subset_name: str, img_code: int | str):
    sub_dir = os.path.join(OUTPUT_DIR, subset_name)
    img_code = prepare_img_code(img_code)
    img_path = os.path.join(sub_dir, "images", img_code + ".jpg")
    info_path = os.path.join(sub_dir, "img_info", img_code + ".json")
    return os.path.exists(img_path) and os.path.exists(info_path)


def resize_img(img: Image.Image, new_width: int, new_height: int):
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def get_img_width_height(img: Image.Image | np.ndarray) -> tuple[int, int]:
    """
    :param img: image to get width and height from
    :return: width, height
    """
    if isinstance(img, Image.Image):
        return img.size
    elif isinstance(img, np.ndarray):
        return img.shape[1], img.shape[0]
    else:
        raise TypeError("Input must be either a PIL Image or a numpy array")


def square_bbox(bbox, image_shape):
    """
    Make the bounding box square by extending the shorter side to match the longer side.
    :param bbox: [x1, y1, x2, y2] - upper left and bottom right corners of the bounding box
    :param image_shape: (height, width) - shape of the image
    :return: bbox: [x1, y1, x2, y2] - upper left and bottom right corners of the bounding box, the box is a square
    """
    bbox_new = bbox.copy()
    mid_x = (bbox[0] + bbox[2]) // 2
    mid_y = (bbox[1] + bbox[3]) // 2
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    m = max(width, height)
    bbox_new[0] = mid_x - m // 2
    bbox_new[2] = mid_x + m // 2
    bbox_new[1] = mid_y - m // 2
    bbox_new[3] = mid_y + m // 2
    image_h = image_shape[0]
    image_w = image_shape[1]
    if bbox_new[0] < 0:
        bbox_new[2] -= bbox_new[0]
        bbox_new[0] = 0
    if bbox_new[1] < 0:
        bbox_new[3] -= bbox_new[1]
        bbox_new[1] = 0
    if bbox_new[2] > image_w:
        bbox_new[0] -= bbox_new[2] - image_w
        bbox_new[2] = image_w
    if bbox_new[3] > image_h:
        bbox_new[1] -= bbox_new[3] - image_h
        bbox_new[3] = image_h
    return bbox_new


def is_valid_bbox(bbox, img_shape):
    """
    Check if the bounding box is valid.
    :param bbox: [x1, y1, x2, y2] - upper left and bottom right corners of the bounding box
    :param img_shape: [height, width] - shape of the image
    :return: bool - True if the bounding box is valid, False otherwise
    """
    if (
        bbox[0] < 0
        or bbox[1] < 0
        or bbox[2] > img_shape[1]
        or bbox[3] > img_shape[0]
        or bbox[0] >= bbox[2]
        or bbox[1] >= bbox[3]
    ):
        return False
    return True


def pad_bbox(bbox, img_shape, pad=0.2):
    """
    Pad the bounding box by a factor of pad.
    :param bbox: [x1, y1, x2, y2] - left upper and right bottom corners of the bounding box
    :param img_shape: (height, width) - shape of the image
    :param pad: float - padding factor
    :return: bbox: [x1, y1, x2, y2] - upper left and bottom right corners of the bounding box, the box is padded
    """
    bbox_new = bbox.copy()
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    pad_x = int(width * pad)
    pad_y = int(height * pad)
    bbox_new[0] = max(0, bbox[0] - pad_x)
    bbox_new[1] = max(0, bbox[1] - pad_y)
    bbox_new[2] = min(bbox[2] + pad_x, img_shape[1])
    bbox_new[3] = min(bbox[3] + pad_y, img_shape[0])
    return bbox_new


def get_bbox_size(bbox):
    """
    Get the size of the bounding box.
    :param bbox: [x1, y1, x2, y2] - upper left and bottom right corners of the bounding box
    :return: size: int - size of the bounding box in pixels
    """
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def sort_faces_by_size(face_info):
    # Create a list of tuples containing face info and their original index
    indexed_faces = [(i, face) for i, face in enumerate(face_info)]

    # Sort by the size of the bounding box (width * height)
    sorted_faces = sorted(
        indexed_faces,
        key=lambda x: (x[1]["bbox"][2] - x[1]["bbox"][0])
        * (x[1]["bbox"][3] - x[1]["bbox"][1]),
        reverse=True,
    )

    # Separate the indices and sorted face info
    sorted_indices = [i for i, face in sorted_faces]
    sorted_face_info = [face for i, face in sorted_faces]

    return sorted_face_info, sorted_indices


def get_identity_img(person_id_code: str) -> np.ndarray:
    """read the identity image from the identities folder and return numpy image"""
    img_path = os.path.join(
        OUTPUT_DIR,
        "identities",
        "images_224x224",
        f"{person_id_code}.png",
    )
    img = np.array(Image.open(img_path))
    return img


def get_img_from_bbox(
    target_image: np.ndarray,
    bbox: list,
    square: bool = False,
    pad: float = 0.0,
    resize: tuple[int, int] | int = None,
):
    """
    Get the image from the bounding box.
    :param target_image: target image, (H, W, C) RGB format
    :param bbox: bounding box in the image
    :param square: should you force the bbox to be square
    :param pad: percentage of padding to add to the bounding box
    :param resize: resize the final bounding box to this dimension
    :return:
    """
    img_shape = target_image.shape[:2]
    bbox = pad_bbox(bbox, img_shape, pad)
    if square:
        bbox = square_bbox(bbox, img_shape)
    if not is_valid_bbox(bbox, img_shape):
        raise ValueError("Invalid bounding box")
    x1, y1, x2, y2 = bbox
    face_img = target_image[y1:y2, x1:x2]
    if resize is not None:
        if isinstance(resize, tuple):
            # If resize is a tuple, use it directly
            face_img = cv2.resize(face_img, resize)
        else:
            # If resize is not a tuple, assume it's an integer
            # Calculate the scaling factor to make the shortest side equal to 'resize'
            h, w = face_img.shape[:2]
            if h < w:
                new_h = resize
                new_w = int(w * (resize / h))
            else:
                new_w = resize
                new_h = int(h * (resize / w))
            face_img = cv2.resize(face_img, (new_w, new_h))
    return face_img
