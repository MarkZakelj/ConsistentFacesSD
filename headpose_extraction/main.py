import argparse

from loguru import logger
from tqdm import tqdm

from headpose_extraction.sixdrepnet import SixDRepNet
from utils import list_directories_in_directory
from utils.imgs import (
    get_face_imgs,
    get_img_info,
    get_number_of_images,
    update_img_info,
)
from utils.paths import OUTPUT_DIR

FORCE = False

model = SixDRepNet(gpu_id=-1)

skip_subsets = ["identities"]


def extract_headpose(face_img):
    pitch, yaw, roll = model.predict(face_img)
    if len(pitch) > 1 or len(yaw) > 1 or len(roll) > 1:
        raise ValueError("More than one face detected in image")

    return pitch[0], yaw[0], roll[0]


def extract_all_headposes_on_subset(subset_name: str):
    if subset_name in skip_subsets:
        return
    logger.info("SUBSET NAME: " + subset_name)

    n_images = get_number_of_images(subset_name)
    for i in tqdm(range(n_images)):
        info = get_img_info(subset_name, i)
        face_imgs = get_face_imgs(subset_name, i)
        for k, face_img in enumerate(face_imgs):
            pitch, yaw, roll = extract_headpose(face_img)
            info["face_info"][k]["head_pose"] = {
                "pitch": float(pitch),
                "yaw": float(yaw),
                "roll": float(roll),
            }
        update_img_info(subset_name, i, info)


def main():
    parser = argparse.ArgumentParser(
        description="HeadPose extraction for a given subset of images."
    )
    parser.add_argument(
        "--subset-name",
        type=str,
        help="The name of the subset to process images for.",
        required=False,
    )
    args = parser.parse_args()
    subset_name = args.subset_name
    subsets = list_directories_in_directory(OUTPUT_DIR)
    if subset_name is not None:
        subsets = [subset_name]
    for subset in subsets:
        extract_all_headposes_on_subset(subset)


if __name__ == "__main__":
    main()
