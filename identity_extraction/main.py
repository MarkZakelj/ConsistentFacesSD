import argparse

import clip
import torch
from loguru import logger
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from face_similarity import FaceEmbedDistance, InsightFace, NoFilterCriteriaError
from image_creation.prompt_construction import construct_identity_prompt_from_code
from utils import list_directories_in_directory
from utils.imgs import (
    get_face_imgs,
    get_identity_img,
    get_img,
    get_img_from_bbox,
    get_img_info,
    get_number_of_images,
    get_prompt_seed_pairs,
    get_subset_info,
    sort_faces_by_size,
    update_img_info,
)
from utils.paths import OUTPUT_DIR

insightFace = InsightFace()

device = "mps"
FORCE = False

model, preprocess = clip.load("ViT-B/32", device=device)

skip_subsets = ["identities"]

all_embedds = {}


def embedd_text_description(text: str):
    if text not in all_embedds:
        print("not in cache")
        text_tokenized = clip.tokenize([text]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokenized)
        all_embedds[text] = text_features
    return all_embedds[text]


def tokenize_text_description(text: str):
    return clip.tokenize([text]).to(device)


def delete_info(subset_name: str):
    if subset_name in skip_subsets:
        return
    logger.info("SUBSET NAME: " + subset_name)

    subset_info = get_subset_info(subset_name)
    n_people = len(subset_info["person_codes"])
    prompt_seed_pairs = get_prompt_seed_pairs(n_people)

    n_images = get_number_of_images(subset_name)
    for i in tqdm(range(n_images)):
        info = get_img_info(subset_name, i)
        if "face_info" not in info:
            continue
        for face_info in info["face_info"]:
            face_info.pop("identity", None)
        update_img_info(subset_name, i, info)


def label_left_to_right(subset_name: str):
    if subset_name in skip_subsets:
        return
    logger.info("SUBSET NAME: " + subset_name)

    subset_info = get_subset_info(subset_name)
    n_people = len(subset_info["person_codes"])
    prompt_seed_pairs = get_prompt_seed_pairs(n_people)

    n_images = get_number_of_images(subset_name)
    for i in tqdm(range(n_images)):
        info = get_img_info(subset_name, i)
        if "face_info" not in info:
            continue
        person_codes = list(prompt_seed_pairs[i]["people"].values())
        face_info = sorted(
            [face_info for face_info in info["face_info"]],
            key=lambda x: (x["bbox"][0] + x["bbox"][2]) / 2,
        )

        for k, face in enumerate(face_info):
            if k >= len(person_codes):
                break
            face["identity"] = person_codes[k]
        update_img_info(subset_name, i, info)


def extract_all_identities_on_subset(subset_name: str):
    if subset_name in skip_subsets:
        return
    logger.info("SUBSET NAME: " + subset_name)

    subset_info = get_subset_info(subset_name)
    n_people = len(subset_info["person_codes"])
    prompt_seed_pairs = get_prompt_seed_pairs(n_people)

    n_images = get_number_of_images(subset_name)
    for i in tqdm(range(n_images)):
        info = get_img_info(subset_name, i)
        if "face_info" not in info:
            continue
        face_info = info["face_info"]
        cont = True
        for face in face_info:
            if "identity" in face and face["identity"] and not FORCE:
                cont = False
                break
        if not cont:
            continue
        person_codes = list(prompt_seed_pairs[i]["people"].values())
        identity_prompts = []
        for person_code in person_codes:
            identity_prompt = construct_identity_prompt_from_code(person_code)
            identity_prompts.append(identity_prompt)
        tokens = clip.tokenize(identity_prompts).to(device)
        faces_bbox, indices = sort_faces_by_size(info["face_info"])
        face_imgs = get_face_imgs(subset_name, i)
        all_probs = []
        for idx in indices[:n_people]:
            face = face_imgs[idx]
            img = preprocess(Image.fromarray(face)).unsqueeze(0).to(device)
            with torch.no_grad():
                logits_per_image, logits_per_text = model(img, tokens)
                probs = -logits_per_image.softmax(dim=-1).cpu().numpy()
                all_probs.append(probs[0].tolist())
        rows, cols = linear_sum_assignment(all_probs)
        mapping = list(zip(rows, cols))
        mapping = sorted(mapping, key=lambda x: x[0])  # not sure if needed
        full_mapping = [-1] * n_people
        for a, b in mapping:
            full_mapping[a] = b
        for k, idx in enumerate(indices[:n_people]):
            info["face_info"][idx]["identity"] = person_codes[full_mapping[k]]
        update_img_info(subset_name, i, info)


def extract_using_embedd_distance(subset_name: str):
    if subset_name in skip_subsets:
        return
    logger.info("SUBSET NAME: " + subset_name)

    subset_info = get_subset_info(subset_name)
    n_people = len(subset_info["person_codes"])
    prompt_seed_pairs = get_prompt_seed_pairs(n_people)
    fed = FaceEmbedDistance()
    n_images = get_number_of_images(subset_name)
    for i in tqdm(range(n_images)):
        info = get_img_info(subset_name, i)
        if "face_info" not in info:
            continue
        img = get_img(subset_name, i)
        face_info = info["face_info"]
        id_codes = list(prompt_seed_pairs[i]["people"].values())
        id_images = [get_identity_img(id_code) for id_code in id_codes]
        scores = []

        for face in face_info:
            face_scores = []
            target_img = get_img_from_bbox(img, face["bbox"], square=False, pad=0.1)
            for id_img in id_images:
                try:
                    dist = fed.analize(
                        insightFace,
                        torch.from_numpy(id_img).unsqueeze(0),
                        torch.from_numpy(target_img).unsqueeze(0),
                        "cosine",
                        filter_thresh=1.0,
                        filter_best=0,
                    )
                    face_scores.append(dist[0])
                except NoFilterCriteriaError:
                    face_scores.append(1.0)
            scores.append(face_scores)
        rows, cols = linear_sum_assignment(scores)
        for r, c in zip(rows, cols):
            try:
                face_info[r]["identity"] = id_codes[c]
            except KeyError as e:
                print(id_codes)
                print(face_info)
                print("Error", r, c, len(face_info), len(id_codes), i)
        update_img_info(subset_name, i, info)


def main():
    parser = argparse.ArgumentParser(
        description="Identity extraction for a given subset of images."
    )
    parser.add_argument(
        "--subset-name",
        type=str,
        help="The name of the subset to process images for.",
        required=False,
    )
    parser.add_argument("--use-first-bbox", action="store_true", required=False)
    parser.add_argument("--delete", action="store_true", required=False)
    args = parser.parse_args()
    subset_name = args.subset_name
    subsets = list_directories_in_directory(OUTPUT_DIR)
    if subset_name is not None:
        if args.delete:
            delete_info(subset_name)
            return
        subsets = [subset_name]
    if args.delete:
        print("You need to specify a subset to delete identities from.")
        return
    for subset in subsets:
        if args.use_first_bbox:
            label_left_to_right(subset)
        else:
            extract_all_identities_on_subset(subset)
            # extract_using_embedd_distance(subset)


if __name__ == "__main__":
    main()
