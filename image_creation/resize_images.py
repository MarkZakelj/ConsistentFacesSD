import os

from PIL import Image

from utils.imgs import get_image_paths, resize_img
from utils.paths import OUTPUT_DIR


def main():
    subset_name = "identities"
    new_size = (224, 224)
    new_img_folder = os.path.join(
        OUTPUT_DIR, subset_name, f"images_{new_size[0]}x{new_size[1]}"
    )

    os.makedirs(new_img_folder, exist_ok=True)

    for img_path in get_image_paths(subset_name):
        img = Image.open(img_path)
        img_resized = resize_img(img, *new_size)
        png_base_name = str(os.path.basename(img_path).replace(".jpg", ".png"))
        img_resized.save(os.path.join(new_img_folder, png_base_name))


if __name__ == "__main__":
    main()
