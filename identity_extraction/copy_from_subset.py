import argparse

from loguru import logger

from utils import list_directories_in_directory
from utils.imgs import get_img_info, get_number_of_images, update_img_info
from utils.paths import OUTPUT_DIR

FORCE = False


def copy_identities_from_subset(from_subset: str, to_subset: str):
    logger.warning(
        "This will overwrite existing identities and remove all other identities"
    )
    n_images_from = get_number_of_images(from_subset)
    n_images_to = get_number_of_images(to_subset)
    if n_images_from != n_images_to:
        logger.error(f"Number of images in {from_subset} and {to_subset} do not match.")
        return
    for i in range(n_images_from):
        info_from = get_img_info(from_subset, i)
        info_to = get_img_info(to_subset, i)
        if "face_info" in info_to and not FORCE:
            logger.warning(f"Already in image {i} of {to_subset}, skipping")
            continue
        if "face_info" not in info_from:
            logger.warning(f"No face info in image {i} of {from_subset}")
            continue
        info_to["face_info"] = info_from["face_info"]
        update_img_info(to_subset, i, info_to)


def main():
    parser = argparse.ArgumentParser(
        description="Pose extraction for a given subset of images."
    )
    parser.add_argument(
        "--from-subset",
        type=str,
        help="The name of the subset to process data from.",
        required=True,
    )
    parser.add_argument(
        "--to-subset",
        type=str,
        help="The name of the subset to process images for.",
        required=True,
    )

    args = parser.parse_args()
    from_subset = args.from_subset
    to_subset = args.to_subset
    all_subsets = list_directories_in_directory(OUTPUT_DIR)
    if from_subset not in all_subsets:
        logger.error(f"Subset {from_subset} not found.")
        return
    if to_subset not in all_subsets:
        logger.error(f"Subset {to_subset} not found.")
        return
    copy_identities_from_subset(from_subset, to_subset)


if __name__ == "__main__":
    main()
