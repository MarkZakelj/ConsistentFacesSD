import argparse

from loguru import logger
from tqdm import tqdm

from utils import list_directories_in_directory
from utils.imgs import get_img_info, get_number_of_images, update_img_info
from utils.paths import OUTPUT_DIR

FORCE = False

skip_subsets = ["identities"]


def extract_number_of_people_front(subset_name: str):
    if subset_name in skip_subsets:
        return
    logger.info("SUBSET NAME: " + subset_name)

    n_images = get_number_of_images(subset_name)
    for i in tqdm(range(n_images)):
        info = get_img_info(subset_name, i)
        if "face_info" not in info:
            n_faces = 0
        else:
            n_faces = len(info["face_info"])
            # raise ValueError("No face info in image")
        # img = get_img(subset_name, i)
        # w, h = get_img_width_height(img)
        # pose_info = model.detect_poses(img, verbose=False)
        # n_poses = 0
        # if pose_info is not None:
        #     people = encode_poses_as_dict(pose_info, h, w)['people']
        #     n_poses = len(people)
        # if n_faces > n_poses:
        #     n_poses = n_faces
        info["n_people_front"] = n_faces
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force the extraction of head poses.",
        required=False,
    )
    args = parser.parse_args()
    if args.force:
        global FORCE
        FORCE = True
    subset_name = args.subset_name
    subsets = list_directories_in_directory(OUTPUT_DIR)
    if subset_name is not None:
        subsets = [subset_name]
    for subset in subsets:
        extract_number_of_people_front(subset)


if __name__ == "__main__":
    main()
