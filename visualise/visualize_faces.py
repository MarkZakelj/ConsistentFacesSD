"""
Show images with bboxes around images
"""

import argparse

import cv2

from utils.imgs import get_image_and_info, get_number_of_images


def main():
    parser = argparse.ArgumentParser(
        description="Face Detection for a given subset of images."
    )
    parser.add_argument("subset_name", type=str, help="The name of the subset")
    args = parser.parse_args()
    subset_name = args.subset_name

    n_images = get_number_of_images(subset_name)
    for i in range(n_images):
        img, info = get_image_and_info(subset_name, i)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if "face_info" in info:
            bbox = info["face_info"]["bbox"]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
