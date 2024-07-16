import json
import os

import streamlit as st
from PIL import Image, ImageDraw

from utils import list_directories_in_directory
from utils.paths import OUTPUT_DIR

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


def draw_bounding_box(image, face_infos):
    draw = ImageDraw.Draw(image)
    for i, info in enumerate(face_infos):
        bbox = info["bbox"]
        draw.rectangle(bbox, outline=colors[i % len(colors)], width=3)
    return image


def main():
    st.title("Image Dataset Visualizer")
    # Number of columns selection
    num_columns = st.sidebar.radio("Number of columns", [1, 2, 3])

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

            # Show prompts option
            show_prompts = st.checkbox(
                f"Show prompts for column {i + 1}", key=f"show_prompts_{i}"
            )

            # Show face bounding boxes option
            show_bboxes = st.checkbox(
                f"Show face bounding boxes for column {i + 1}", key=f"show_bboxes_{i}"
            )

            if dataset:
                dataset_path = config[dataset]
                images_path = os.path.join(dataset_path, "images")
                img_info_path = os.path.join(dataset_path, "img_info")

                # Read and display info.json
                info_json_path = os.path.join(dataset_path, "info.json")
                if os.path.exists(info_json_path):
                    info = load_json(info_json_path)
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

                for img_file in images_to_display:
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
                        img = draw_bounding_box(img, img_info["face_info"])

                    with st.container():
                        st.image(img, use_column_width=True)
                        if show_prompts and img_info:
                            st.write(f"Prompt: {img_info.get('prompt', 'N/A')}")


if __name__ == "__main__":
    main()
