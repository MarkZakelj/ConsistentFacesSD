import json
import os

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from face_similarity.main import calculate_similarity
from headpose_extraction.sixdrepnet.utils import plot_pose_cube
from pose_extraction.dwpose import decode_json_as_poses
from pose_extraction.dwpose.util import draw_bodypose, draw_facepose
from utils import list_directories_in_directory
from utils.imgs import pad_bbox, square_bbox, update_img_info
from utils.paths import OUTPUT_DIR

font = ImageFont.truetype("Arial", size=19)
font_small = ImageFont.truetype("Arial", size=18)
# Configuration dictionary (you can modify this as needed)
config = {
    str(setname): str(os.path.join(OUTPUT_DIR, setname))
    for setname in list_directories_in_directory(OUTPUT_DIR)
}

st.set_page_config(layout="wide")


def load_image(image_path):
    return Image.open(image_path)


def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


colors = ["red", "green", "blue", "black"]


def draw_bounding_box(
    image,
    face_infos,
    show_identity=False,
    show_number=False,
    identity_similarity=False,
    face_quality=False,
):
    draw = ImageDraw.Draw(image)
    image_size = (image.size[1], image.size[0])
    for i, info in enumerate(face_infos):
        bbox = info["bbox"]
        bbox = pad_bbox(bbox, image_size, 0.2)
        bbox = square_bbox(bbox, image_size)
        draw.rectangle(bbox, outline=colors[i % len(colors)], width=3)
        if show_number:
            draw.text(
                (bbox[0], bbox[1] - 20), f"{i + 1}", fill="white", font=font_small
            )
        all_text = ""
        identity = None
        if show_identity:
            text = info.get("identity", "unknown")
            if not text:
                text = "unknown"
            else:
                identity = text
            all_text += text
            fill = "gray" if text == "unknown" else "white"
            if not identity_similarity:
                draw.text((bbox[0], bbox[1] - 13), text, fill=fill, font=font)
        if identity_similarity:
            text = f"{info.get('similarity_cosine', -1):.2f}"
            show = info.get("similarity_cosine", -1) > 0.0
            fill = "white" if show else "gray"
            all_text += f" {text}"
            position = (bbox[0], bbox[1] - 16)
            left, top, right, bottom = draw.textbbox(position, all_text, font=font)
            if identity:
                draw.rectangle((left - 2, top - 2, right + 2, bottom), fill="black")
                draw.text(position, all_text, fill=fill, font=font)
        if face_quality:
            text = f"{info.get('face_quality', -1):.2f}"
            position = (bbox[0], bbox[3])
            left, top, right, bottom = draw.textbbox(position, all_text, font=font)
            draw.rectangle((left - 2, top - 2, right + 2, bottom), fill="black")
            draw.text(position, text, fill="white", font=font)

    return image


def draw_poses(image, pose_infos):
    poses, _, _, _ = decode_json_as_poses(pose_infos)
    canvas = np.array(image)
    for pose in poses:
        canvas = draw_bodypose(canvas, pose.body.keypoints)
        canvas = draw_facepose(canvas, pose.face)
    return Image.fromarray(canvas)


def draw_img_num(image, img_num):
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), f"{img_num}", fill="white", font=font)
    return image


def draw_clip_score(image, score):
    draw = ImageDraw.Draw(image)
    w, h = image.size
    draw.text((w - 50, h - 22), f"{score}", fill="white", font=font)
    return image


def draw_headposes(image, face_infos):
    image = np.array(image)
    for i, info in enumerate(face_infos):
        if "head_pose" not in info:
            continue
        bbox = info["bbox"]
        head_pose = info["head_pose"]
        x_min = int(bbox[0])
        y_min = int(bbox[1])
        x_max = int(bbox[2])
        y_max = int(bbox[3])
        bbox_width = abs(x_max - x_min)
        bbox_height = abs(y_max - y_min)

        x_min = max(0, x_min - int(0.2 * bbox_height))
        y_min = max(0, y_min - int(0.2 * bbox_width))
        x_max = x_max + int(0.2 * bbox_height)
        y_max = y_max + int(0.2 * bbox_width)
        yaw = head_pose["yaw"]
        pitch = head_pose["pitch"]
        roll = head_pose["roll"]
        image = plot_pose_cube(
            image,
            yaw,
            pitch,
            roll,
            x_min + int(0.5 * (x_max - x_min)),
            y_min + int(0.5 * (y_max - y_min)),
            size=bbox_width,
        )
    return Image.fromarray(image)


def main():
    st.title("Image Dataset Visualizer")
    # Number of columns selection
    num_columns = st.sidebar.radio("Number of columns", [1, 2, 3], index=1)

    # Image limit selection (applies to all columns)
    image_limit = st.sidebar.selectbox(
        "Number of images to display per column", [10, 50, 100], key="image_limit"
    )

    # Pagination control (applies to all columns)
    total_images = max(
        len(files)
        for files in [
            sorted(
                [
                    f
                    for f in os.listdir(os.path.join(config[dataset], "images"))
                    if f.endswith(".jpg")
                ]
            )
            for dataset in config.keys()
        ]
    )
    max_page = (total_images - 1) // image_limit
    page_number = st.sidebar.number_input(
        "Select page",
        min_value=0,
        max_value=max_page,
        value=0,
        step=1,
        key="page_number",
    )

    # Show prompts option
    show_prompts = st.sidebar.checkbox("Show prompts", key="show_prompts")
    show_number = st.sidebar.checkbox("Show image numbers", key="show_number")

    show_clip_scores = st.sidebar.checkbox("Show clip scores", key="show_clip_scores")

    show_face_quality = st.sidebar.checkbox(
        "Show face quality", key="show_face_quality"
    )
    show_headposes = st.sidebar.checkbox("Show headposes", key="show_headposes")
    # Show face bounding boxes option
    show_bboxes = st.sidebar.checkbox("Show face bounding boxes", key="show_bboxes")
    show_identity = st.sidebar.checkbox("Show identity", key="show_identity")
    show_identity_similarity = st.sidebar.checkbox(
        "Show identity similarity", key="show_identity_similarity"
    )

    show_bbox_num = st.sidebar.checkbox(
        "Show bounding box numbers", key="show_bbox_num"
    )
    change_face_identities = st.sidebar.checkbox(
        "Change face identities", key="change_face_identities"
    )
    change_n_people_front = st.sidebar.checkbox(
        "Change n_people_front", key="change_n_people_front"
    )
    only_show_off_people = st.sidebar.checkbox(
        "Only show off people", key="only_show_off_people"
    )
    show_seed = st.sidebar.checkbox("Show seed", key="show_seed")
    # Create columns based on user selection
    columns = st.columns(num_columns)

    for i in range(num_columns):
        with columns[i]:
            st.subheader(f"Column {i + 1}")

            # Dataset selection
            dataset = st.selectbox(
                f"Select dataset for column {i + 1}",
                sorted(list(config.keys())),
                key=f"dataset_{i}",
            )

            if dataset:
                dataset_path = config[dataset]
                images_path = os.path.join(dataset_path, "images")
                img_info_path = os.path.join(dataset_path, "img_info")

                # Read and display info.json
                info_json_path = os.path.join(dataset_path, "info.json")
                n_target_people = -1
                if os.path.exists(info_json_path):
                    info = load_json(info_json_path)
                    n_target_people = len(info.get("person_codes", []))
                    with st.expander("Dataset Info:"):
                        for key, value in info.items():
                            st.write(f"{key}: {value}")

                # Get list of image files
                image_files = sorted(
                    [f for f in os.listdir(images_path) if f.endswith(".jpg")]
                )

                # Calculate start and end indices
                start_index = page_number * image_limit
                end_index = start_index + image_limit
                images_to_display = image_files[start_index:end_index]

                for n, img_file in enumerate(images_to_display):
                    img_num = start_index + n
                    img_path = os.path.join(images_path, img_file)
                    img = load_image(img_path)

                    json_file = img_file.replace(".jpg", ".json")
                    json_path = os.path.join(img_info_path, json_file)
                    img_info = {}

                    if os.path.exists(json_path):
                        try:
                            img_info = load_json(json_path)
                        except json.decoder.JSONDecodeError:
                            print(json_path)
                            raise ValueError

                    if show_bboxes and "face_info" in img_info:
                        img = draw_bounding_box(
                            img,
                            img_info["face_info"],
                            show_identity=show_identity,
                            show_number=show_bbox_num,
                            identity_similarity=show_identity_similarity,
                            face_quality=show_face_quality,
                        )

                    # if show_poses and "pose_info" in img_info:
                    #     img = draw_poses(img, img_info["pose_info"])

                    if show_headposes and "face_info" in img_info:
                        img = draw_headposes(img, img_info["face_info"])
                    if show_number:
                        img = draw_img_num(img, img_num)
                    if show_clip_scores:
                        clip_score = img_info.get("clip_score", {}).get(
                            "openai/clip-vit-base-patch16", 0
                        )
                        img = draw_clip_score(img, clip_score)
                    with st.container():
                        if only_show_off_people:
                            if img_info.get("n_people_front", 0) != n_target_people:
                                st.image(img, use_column_width=True)
                        else:
                            st.image(img, use_column_width=True)
                        if show_prompts and img_info:
                            st.write(f"Prompt: {img_info.get('prompt', 'N/A')}")
                        if show_seed and img_info:
                            st.write(f"Seed: {img_info.get('seed', 'N/A')}")

                        if change_n_people_front:
                            n_people_front = st.number_input(
                                "Number of people in front",
                                value=img_info.get("n_people_front", 0),
                                key=f"n_people_front_{img_num}_{dataset_path}",
                            )
                            img_info["n_people_front"] = n_people_front
                            if st.button(
                                "Save n_people_front",
                                key=f"save_n_people_front_{img_num}_{dataset_path}",
                            ):
                                update_img_info(dataset, img_num, img_info)
                                st.rerun()

                        if change_face_identities:
                            for j, info in enumerate(img_info.get("face_info", [])):
                                new_identity = st.text_input(
                                    f"Identity {j + 1}",
                                    info.get("identity", ""),
                                    key=f"identity_{img_num}_{dataset_path}_{j}_{i}",
                                )
                                info["identity"] = new_identity

                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(
                                    "Save identities",
                                    key=f"save_identities_{img_num}_{dataset_path}_{i}",
                                ):
                                    for j, info in enumerate(
                                        img_info.get("face_info", [])
                                    ):
                                        if "similarity_cosine" in info:
                                            del info["similarity_cosine"]
                                    update_img_info(dataset, img_num, img_info)
                                    st.rerun()

                            with col2:
                                if st.button(
                                    "Recalculate ID",
                                    key=f"recalculate_identities_{img_num}_{dataset_path}_{i}",
                                ):
                                    calculate_similarity(dataset, ids=[img_num])
                                    st.rerun()


if __name__ == "__main__":
    main()
