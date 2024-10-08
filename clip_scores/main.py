import argparse

import torch
from loguru import logger
from torchmetrics.multimodal.clip_score import CLIPScore
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
clip_version = "openai/clip-vit-base-patch16"
metric = CLIPScore(model_name_or_path=clip_version)

generator = torch.manual_seed(42)


def calculate_clip_scores_on_subset(subset_name: str):
    if subset_name in skip_subsets:
        return
    logger.info("SUBSET NAME: " + subset_name)

    n_images = get_number_of_images(subset_name)
    for i in tqdm(range(n_images)):
        info = get_img_info(subset_name, i)
        if "clip_score" in info and clip_version in info["clip_score"] and not FORCE:
            continue
        img = get_img(subset_name, i)
        img = np_2_tensor(img)
        prompt = info["prompt"]
        prompt_clean = (
            prompt.replace("(cinematic still)", "photo")
            .replace("-", " ")
            .replace("(", "")
        ).replace(")", "")
        prompt_clean = prompt_clean.split(".")[0]
        clip_score = metric(img, prompt_clean)
        clip_score = clip_score.detach().cpu().numpy().item()
        if "clip_score" not in info:
            info["clip_score"] = {}
        info["clip_score"][clip_version] = clip_score
        update_img_info(subset_name, i, info)


def main():
    parser = argparse.ArgumentParser(
        description="CLIP score calculation for a given subset of images."
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
        help="Force recompute CLIP score for all images.",
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
        calculate_clip_scores_on_subset(subset)


if __name__ == "__main__":
    main()
