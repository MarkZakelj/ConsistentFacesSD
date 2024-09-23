import argparse

import torch
from loguru import logger
from tqdm import tqdm

from face_similarity import FaceEmbedDistance, InsightFace, NoFilterCriteriaError
from utils import list_directories_in_directory
from utils.imgs import (
    get_identity_img,
    get_img,
    get_img_from_bbox,
    get_img_info,
    get_number_of_images,
    update_img_info,
)
from utils.paths import OUTPUT_DIR

FORCE = False

skip_subsets = ["identities"]

insightFace = InsightFace()


def calculate_similarity(subset_name: str, ids=None):
    if subset_name in skip_subsets:
        return
    logger.info("SUBSET NAME: " + subset_name)

    n_images = get_number_of_images(subset_name)
    fed = FaceEmbedDistance()
    if ids is None:
        ids = range(n_images)
    for i in tqdm(ids):
        info = get_img_info(subset_name, i)
        img = get_img(subset_name, i)
        if "face_info" not in info:
            continue
        face_info = info["face_info"]
        for face in face_info:
            identity = face.get("identity")
            if not identity:
                continue
            if "similarity_cosine" in face and not FORCE:
                continue
            identity_img = get_identity_img(identity)
            target_img = get_img_from_bbox(
                img, face["bbox"], square=False, pad=0.1, resize=256
            )
            try:
                dist = fed.analize(
                    insightFace,
                    torch.from_numpy(identity_img).unsqueeze(0),
                    torch.from_numpy(target_img).unsqueeze(0),
                    "cosine",
                    filter_thresh=1.0,
                    filter_best=0,
                )
                if len(dist) > 1:
                    logger.error(f"More than one distance returned for image {i}")
                face["similarity_cosine"] = 1 - dist[0]
            except NoFilterCriteriaError:
                logger.warning(f"No filter criteria for image {i}")
        update_img_info(subset_name, i, info)


def main():
    parser = argparse.ArgumentParser(
        description="Face similarity calculation for a given subset of images."
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
        help="Force the recalculation of similarity for all images.",
        required=False,
    )
    args = parser.parse_args()
    subset_name = args.subset_name
    all_subsets = list_directories_in_directory(OUTPUT_DIR)
    if subset_name is not None:
        subsets = [subset_name]
    else:
        subsets = all_subsets
    if args.force and args.subset_name is not None:
        global FORCE
        FORCE = True
    for subset in subsets:
        if subset not in all_subsets:
            logger.error(f"Subset {subset} does not exist.")
            continue
        calculate_similarity(subset)


if __name__ == "__main__":
    main()
