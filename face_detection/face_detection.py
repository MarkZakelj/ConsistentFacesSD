import argparse

import numpy as np
from insightface.app import FaceAnalysis
from loguru import logger
from tqdm import tqdm

from utils import list_directories_in_directory
from utils.imgs import get_img, get_img_info, get_number_of_images, update_img_info
from utils.paths import OUTPUT_DIR

FORCE = False

app = FaceAnalysis(allowed_modules=["detection"])
app.prepare(ctx_id=0, det_size=(640, 640))

skip_subsets = ["identities"]


def detect_face(img):
    faces = app.get(img)
    if len(faces) == 0:
        print("No face detected in image")
        return None
    return faces


def delete_face_info(subset_name: str):
    if subset_name in skip_subsets:
        return
    logger.info("SUBSET NAME: " + subset_name)

    n_images = get_number_of_images(subset_name)
    for i in tqdm(range(n_images)):
        info = get_img_info(subset_name, i)
        if "face_info" in info:
            del info["face_info"]
            update_img_info(subset_name, i, info, replace=True)
        else:
            print(f"No face info in image {i}")


def detect_all_faces_on_subset(subset_name: str):
    if subset_name in skip_subsets:
        return
    logger.info("SUBSET NAME: " + subset_name)

    n_images = get_number_of_images(subset_name)
    for i in tqdm(range(n_images)):
        info = get_img_info(subset_name, i)
        if "face_info" in info and not FORCE:
            continue
        img = get_img(subset_name, i)
        face_infos = detect_face(img)
        if face_infos is None:
            continue
        if "face_info" not in info or FORCE:
            info["face_info"] = []
        for face_info in face_infos:
            face_info.bbox = face_info.bbox.astype(np.int32).tolist()
            face_info.kps = face_info.kps.astype(np.int32).tolist()
            face_info.det_score = float(face_info.det_score)
            info["face_info"].append(
                {
                    "bbox": face_info.bbox,
                    "kps": face_info.kps,
                    "det_score": face_info.det_score,
                }
            )
        update_img_info(subset_name, i, info)


def main():
    parser = argparse.ArgumentParser(
        description="Face Detection for a given subset of images."
    )
    parser.add_argument(
        "--subset-name",
        type=str,
        help="The name of the subset to process images for.",
        required=False,
    )
    parser.add_argument(
        "--force",
        help="force calculation of face info even when it exists",
        action="store_true",
        required=False,
    )
    parser.add_argument("--delete", action="store_true", required=False)
    args = parser.parse_args()
    subset_name = args.subset_name
    global FORCE
    if args.force:
        FORCE = True
    subsets = list_directories_in_directory(OUTPUT_DIR)
    if subset_name is not None:
        if args.delete:
            delete_face_info(subset_name)
            return
        subsets = [subset_name]
    if args.delete:
        print("You need to specify a subset to delete identities from.")
        return
    for subset in subsets:
        detect_all_faces_on_subset(subset)


if __name__ == "__main__":
    main()
