import argparse

import torch
from loguru import logger
from torchmetrics.multimodal import CLIPImageQualityAssessment
from tqdm import tqdm

from utils import list_directories_in_directory
from utils.imgs import (
    get_img,
    get_img_info,
    get_number_of_images,
    np_2_tensor,
    update_img_info,
)
from utils.paths import OUTPUT_DIR

FORCE = False

skip_subsets = ["identities"]
clip_version = "clip_iqa"
PROMPTS = ("quality", "natural")
metric = CLIPImageQualityAssessment(
    model_name_or_path="clip_iqa", data_range=255.0, prompts=PROMPTS
)
DEVICE = "mps"
metric.to(DEVICE)
generator = torch.manual_seed(42)


def calculate_clipiqa_scores_on_subset(subset_name: str):
    if subset_name in skip_subsets:
        return
    logger.info("SUBSET NAME: " + subset_name)

    n_images = get_number_of_images(subset_name)
    for i in tqdm(range(n_images)):
        info = get_img_info(subset_name, i)
        if "clip_iqa" in info and not FORCE:
            continue
        img = get_img(subset_name, i)
        img = np_2_tensor(img)
        img = img.to(DEVICE)
        clipiqa_scores = metric(img)
        info["clip_iqa"] = {
            "quality": clipiqa_scores["quality"].item(),
            "natural": clipiqa_scores["natural"].item(),
        }
        update_img_info(subset_name, i, info)


def main():
    parser = argparse.ArgumentParser(
        description="CLIP-IQA score calculation for a given subset of images."
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
        help="Force recompute CLIP-IQA score for all images.",
        required=False,
    )
    args = parser.parse_args()
    global FORCE
    if args.force:
        FORCE = True
    subset_name = args.subset_name
    subsets = list_directories_in_directory(OUTPUT_DIR)
    if subset_name is not None:
        subsets = [subset_name]
    for subset in subsets:
        calculate_clipiqa_scores_on_subset(subset)


if __name__ == "__main__":
    main()
