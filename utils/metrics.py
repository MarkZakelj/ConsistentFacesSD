import pandas as pd

from utils.imgs import get_img_info, get_number_of_images


def construct_dataframe(subset_name: str):
    pass


def construct_dataframe_faces(subset_name: str):
    n_images = get_number_of_images(subset_name)
    data = []
    for i in range(n_images):
        info = get_img_info(subset_name, i)
        if "face_info" not in info:
            continue
        for face_info in info["face_info"]:
            data.append(
                {
                    "subset": subset_name,
                    "image_id": i,
                    "bbox": face_info["bbox"],
                    "kps": face_info["kps"],
                    "det_score": face_info["det_score"],
                    "face_quality": face_info["face_quality"],
                }
            )
    df = pd.DataFrame(data)
    return df
