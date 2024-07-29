import argparse

from loguru import logger
from tqdm import tqdm

from pose_extraction import get_dw_pose_model
from pose_extraction.dwpose import encode_poses_as_dict
from utils import list_directories_in_directory
from utils.imgs import (
    get_img,
    get_img_info,
    get_img_width_height,
    get_number_of_images,
    update_img_info,
)
from utils.paths import OUTPUT_DIR

FORCE = False

model = get_dw_pose_model()

skip_subsets = ["identities"]


def extract_pose(img):
    return model.detect_poses(img, verbose=False)


def extract_all_poses_on_subset(subset_name: str):
    if subset_name in skip_subsets:
        return
    logger.info("SUBSET NAME: " + subset_name)

    n_images = get_number_of_images(subset_name)
    for i in tqdm(range(n_images)):
        info = get_img_info(subset_name, i)
        if "pose_info" in info and not FORCE:
            continue
        img = get_img(subset_name, i)
        w, h = get_img_width_height(img)
        pose_info = extract_pose(img)
        if pose_info is None:
            continue
        info["pose_info"] = encode_poses_as_dict(pose_info, h, w)
        update_img_info(subset_name, i, info)


def main():
    parser = argparse.ArgumentParser(
        description="Pose extraction for a given subset of images."
    )
    parser.add_argument(
        "--subset_name",
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
        extract_all_poses_on_subset(subset)


if __name__ == "__main__":
    main()
