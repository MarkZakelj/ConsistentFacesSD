import numpy as np
import pandas as pd

from utils.imgs import get_bbox_size, get_img_info, get_number_of_images


def construct_dataframe(subset_name: str):
    n_images = get_number_of_images(subset_name)
    data = []
    for i in range(n_images):
        info = get_img_info(subset_name, i)
        clip_score = info.get("clip_score", {})
        data.append(
            {
                "subset": subset_name,
                "image_id": i,
                "clip_score": clip_score.get("openai/clip-vit-base-patch16"),
                "quality": info.get("clip_iqa", {}).get("quality"),
                "natural": info.get("clip_iqa", {}).get("natural"),
                "n_people_front": info.get("n_people_front", 0),
            }
        )
    df = pd.DataFrame(data)
    return df


def construct_dataframe_faces(subset_name: str):
    n_images = get_number_of_images(subset_name)
    data = []
    for i in range(n_images):
        info = get_img_info(subset_name, i)
        if "face_info" not in info:
            continue
        for face_info in info["face_info"]:
            head_pose = face_info.get("head_pose", {})
            data.append(
                {
                    "subset": subset_name,
                    "image_id": i,
                    "bbox": face_info.get("bbox"),
                    "det_score": face_info.get("det_score"),
                    "face_quality": face_info.get("face_quality"),
                    "identity": face_info.get("identity"),
                    "similarity_cosine": face_info.get("similarity_cosine"),
                    "pitch": head_pose.get("pitch"),
                    "yaw": head_pose.get("yaw"),
                    "roll": head_pose.get("roll"),
                    "clip_score": (
                        info["clip_score"].get("openai/clip-vit-base-patch16")
                        if "clip_score" in info
                        else None
                    ),
                }
            )
    df = pd.DataFrame(data)
    df["bbox_size"] = df["bbox"].apply(lambda x: get_bbox_size(x))
    return df


def get_dataframe_faces(subset_name: str, identity_only: bool = True):
    df_raw = construct_dataframe_faces(subset_name)
    if identity_only:
        df_raw[df_raw["identity"] == ""] = np.nan
        df_raw = df_raw.dropna(subset=["identity"])
    return df_raw


def get_dataframe(subset_name: str):
    df_raw = construct_dataframe(subset_name)
    return df_raw
