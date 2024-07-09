import argparse

import numpy as np
from insightface.app import FaceAnalysis
from loguru import logger
from tqdm import tqdm

from utils.imgs import get_img, get_img_info, get_number_of_images, update_img_info

app = FaceAnalysis(allowed_modules=["detection"])
app.prepare(ctx_id=0, det_size=(640, 640))


def detect_face(img):
    faces = app.get(img, max_num=1)
    if len(faces) == 0:
        print("No face detected in image")
        return None
    return faces[0]


def main():
    parser = argparse.ArgumentParser(
        description="Face Detection for a given subset of images."
    )
    parser.add_argument(
        "subset_name", type=str, help="The name of the subset to process images for."
    )
    args = parser.parse_args()
    subset_name = args.subset_name

    logger.info("SUBSET NAME: " + subset_name)

    n_images = get_number_of_images(subset_name)
    for i in tqdm(range(n_images)):
        info = get_img_info(subset_name, i)
        if "face_info" in info:
            continue
        img = get_img(subset_name, i)
        logger.info(f"Detecting face on image {i}")
        face_info = detect_face(img)
        if face_info is None:
            logger.info("No Face detected!")
            continue
        face_info.bbox = face_info.bbox.astype(np.int32).tolist()
        face_info.kps = face_info.kps.astype(np.int32).tolist()
        face_info.det_score = float(face_info.det_score)
        info["face_info"] = {
            "bbox": face_info.bbox,
            "kps": face_info.kps,
            "det_score": face_info.det_score,
        }
        update_img_info(subset_name, i, info)


if __name__ == "__main__":
    main()
