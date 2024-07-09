import json
import os

script_dir = os.path.dirname(__file__)


def create_test_output_path(script_dir, test_name):
    test_out_path = os.path.join(script_dir, "output", f"{test_name}")
    if not os.path.exists(test_out_path):
        os.makedirs(test_out_path)
    return test_out_path


def save_img_from_comfy(imgs, test_name):
    test_out_path = create_test_output_path(script_dir, test_name)
    for i, img in enumerate(imgs):
        img.save(os.path.join(test_out_path, f"image-{i}.png"))


def save_timings(timings, test_name):
    test_out_path = create_test_output_path(script_dir, test_name)
    with open(os.path.join(test_out_path, "timings.json"), "w") as f:
        json.dump(timings, f, indent=2)


def save_workflow(workflow, test_name):
    test_out_path = create_test_output_path(script_dir, test_name)
    with open(os.path.join(test_out_path, "workflow.json"), "w") as f:
        json.dump(workflow, f, indent=2)
