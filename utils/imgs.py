import base64
from io import BytesIO

from PIL import Image


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
    with open(path, "r") as f:
        base64_string = f.read().strip()
    if as_image:
        return base64_2_img(base64_string)
    return base64_string


def resize_image(image: Image.Image, new_size: int) -> Image.Image:
    original_size = min(image.size)
    if original_size <= new_size:
        return image
    k = new_size / original_size
    return image.resize((int(image.width * k), int(image.height * k)), Image.LANCZOS)
