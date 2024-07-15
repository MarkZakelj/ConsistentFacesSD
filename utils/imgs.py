import base64
import json
import os
from io import BytesIO

import cv2
from PIL import Image

from utils.paths import OUTPUT_DIR


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


def get_img_info(subset_name: str, img_code: int):
    sub_dir = os.path.join(OUTPUT_DIR, subset_name)
    info = json.load(
        open(os.path.join(sub_dir, "img_info", prepare_img_code(img_code) + ".json"))
    )
    return info


def get_img(subset_name: str, img_code: int | str):
    sub_dir = os.path.join(OUTPUT_DIR, subset_name)
    img_path = os.path.join(sub_dir, "images", prepare_img_code(img_code) + ".jpg")
    img = cv2.imread(img_path)[..., [2, 1, 0]]
    return img


def get_image_and_info(subset_name: str, img_code: int | str):
    img = get_img(subset_name, img_code)
    info = get_img_info(subset_name, img_code)
    return img, info


def update_img_info(subset_name: str, img_code: int | str, info_update: dict):
    info = get_img_info(subset_name, img_code)
    info = info | info_update
    sub_dir = os.path.join(OUTPUT_DIR, subset_name)
    info_path = os.path.join(sub_dir, "img_info", prepare_img_code(img_code) + ".json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)


def get_number_of_images(subset_name: str):
    sub_dir = os.path.join(OUTPUT_DIR, subset_name)
    return count_images_in_dir(os.path.join(sub_dir, "images"))


def get_image_paths(subset_name: str, img_folder: str = "images"):
    sub_dir = os.path.join(OUTPUT_DIR, subset_name)
    return get_img_paths_in_dir(os.path.join(sub_dir, img_folder))


def image_exists(subset_name: str, img_code: int | str):
    sub_dir = os.path.join(OUTPUT_DIR, subset_name)
    if isinstance(img_code, int):
        img_code = prepare_img_code(img_code)
    img_path = os.path.join(sub_dir, "images", img_code + ".jpg")
    info_path = os.path.join(sub_dir, "img_info", img_code + ".json")
    return os.path.exists(img_path) and os.path.exists(info_path)


def resize_img(img: Image.Image, new_width: int, new_height: int):
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
