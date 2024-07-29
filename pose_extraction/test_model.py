import os

from __init__ import get_dw_pose_model

from utils.imgs import get_img2

script_dir = os.path.dirname(__file__)


def main():
    # img = Image.open(os.path.join(script_dir, "example_img.jpg"))
    # input_image = np.array(img, dtype=np.uint8)
    # img = get_img("base_one_person", 0)
    model = get_dw_pose_model()
    for i in range(10):
        img = get_img2("base_one_person", i)
        poses = model.detect_poses(img)
        del img
        del poses


if __name__ == "__main__":
    main()
