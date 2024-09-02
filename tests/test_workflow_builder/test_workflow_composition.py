import json
import os
from functools import reduce

from workflow_builder import (
    MultipleCharactersFaceIdWorkflowManager,
    MultipleCharactersWorkflowManager,
    TextToImageWorkflowManager,
    TwoCharactersFaceIdWorkflowManager,
)

from .test_workflow_utils import output_dir

# get directory of current file
current_dir = os.path.dirname(os.path.abspath(__file__))

imgs_path = os.path.join(current_dir, "imgs")

module_nodes = json.load(open(os.path.join(current_dir, "module_nodes.json"), "r"))


def get_workflow_keys(workflow_name: str | dict):
    if isinstance(workflow_name, str):
        return module_nodes[workflow_name]
    else:
        return list(workflow_name.keys())


def merge_lists_into_set(lists: list[list]):
    return reduce(lambda a, b: a | set(b), lists, set())


def create_key_set(workflows: list[str]):
    return merge_lists_into_set([get_workflow_keys(wf) for wf in workflows])


class TestWorkflowManagerNodesConnected:
    def test_basic(self):
        wm = TextToImageWorkflowManager()
        wm.trim_workflow()
        assert set(get_workflow_keys(wm.workflow)) == set(get_workflow_keys("basic"))
        wm.save_workflow(os.path.join(output_dir, "text_to_image", "basic.json"))

    def test_basic_pose(self):
        wm = TextToImageWorkflowManager(["pose"])
        wm.trim_workflow()
        files = ["basic", "pose"]
        assert set(get_workflow_keys(wm.workflow)) == create_key_set(files)
        wm.save_workflow(os.path.join(output_dir, "text_to_image", "pose.json"))

    def test_basic_lora(self):
        wm = TextToImageWorkflowManager(["lora"])
        wm.trim_workflow()
        files = ["basic", "lora"]
        assert set(get_workflow_keys(wm.workflow)) == create_key_set(files)
        wm.save_workflow(os.path.join(output_dir, "text_to_image", "lora.json"))

    def test_pose_lora(self):
        wm = TextToImageWorkflowManager(["pose", "lora"])
        wm.trim_workflow()
        files = ["basic", "lora", "pose"]
        assert set(get_workflow_keys(wm.workflow)) == create_key_set(files)
        wm.save_workflow(os.path.join(output_dir, "text_to_image", "lora_pose.json"))

    def test_face_lora(self):
        wm = TextToImageWorkflowManager(["face", "lora"])
        wm.trim_workflow()
        files = ["basic", "lora", "face"]
        assert set(get_workflow_keys(wm.workflow)) == create_key_set(files)
        wm.save_workflow(os.path.join(output_dir, "text_to_image", "lora_face.json"))

    def test_lora_pose_face(self):
        wm = TextToImageWorkflowManager(["lora", "pose", "face"])
        wm.trim_workflow()
        files = ["basic", "lora", "face", "pose"]
        assert set(get_workflow_keys(wm.workflow)) == create_key_set(files)
        wm.save_workflow(
            os.path.join(output_dir, "text_to_image", "lora_pose_face.json")
        )

    def test_pose_face(self):
        wm = TextToImageWorkflowManager(["pose", "face"])
        wm.trim_workflow()
        files = ["basic", "pose", "face"]
        assert set(get_workflow_keys(wm.workflow)) == create_key_set(files)
        wm.save_workflow(os.path.join(output_dir, "text_to_image", "pose_face.json"))

    def test_character(self):
        wm = TextToImageWorkflowManager(["character"])
        wm.trim_workflow()
        files = ["basic", "character"]
        assert set(get_workflow_keys(wm.workflow)) == create_key_set(files)
        wm.save_workflow(os.path.join(output_dir, "text_to_image", "character.json"))


class TestMultipleCharactersManagerNodesConnected:
    def test_single_character(self):
        wm = MultipleCharactersWorkflowManager(n_characters=1)
        wm.trim_workflow()
        files = ["basic", "multiple_chars", "p0", "pose_gen"]
        assert set(get_workflow_keys(wm.workflow)) == create_key_set(files)
        wm.save_workflow(
            os.path.join(output_dir, "multiple_characters", "one_char.json")
        )

    def test_two_characters(self):
        wm = MultipleCharactersWorkflowManager(n_characters=2, features=["lora"])
        wm.trim_workflow()
        files = ["basic", "multiple_chars", "lora", "p0", "p1", "pose_gen"]
        assert set(get_workflow_keys(wm.workflow)) == create_key_set(files)
        wm.save_workflow(
            os.path.join(output_dir, "multiple_characters", "two_chars_lora.json")
        )

    def test_custom_skeletons(self):
        wm = MultipleCharactersWorkflowManager(
            n_characters=1, features=["lora", "pose"]
        )
        wm.trim_workflow()
        files = ["basic", "multiple_chars", "lora", "mc_pose", "p0", "pose_gen"]
        assert set(get_workflow_keys(wm.workflow)) == create_key_set(files)
        wm.save_workflow(
            os.path.join(output_dir, "multiple_characters", "one_char_lora_pose.json")
        )

    def test_two_characters_lora_pose(self):
        wm = MultipleCharactersWorkflowManager(
            n_characters=2, features=["lora", "pose"]
        )
        wm.trim_workflow()
        files = ["basic", "multiple_chars", "lora", "mc_pose", "p0", "p1", "pose_gen"]
        assert set(get_workflow_keys(wm.workflow)) == create_key_set(files)
        wm.save_workflow(
            os.path.join(output_dir, "multiple_characters", "two_char_pose.json")
        )


class TestTwoCharactersNodesConnected:
    def test_two_characters_faceid(self):
        wm = TwoCharactersFaceIdWorkflowManager()
        wm.trim_workflow()
        files = [
            "basic",
            "lora",
            "faceid",
            "IPAdapterFaceID0",
            "IPAdapterFaceID1",
        ]
        assert set(get_workflow_keys(wm.workflow)) == create_key_set(files)
        wm.save_workflow(os.path.join(output_dir, "two_characters_faceid.json"))

    def test_two_characters_faceid_pose(self):
        wm = TwoCharactersFaceIdWorkflowManager()
        wm.trim_workflow()
        files = [
            "basic",
            "lora",
            "mcfaceid",
            "mcfaceidp0",
            "mcfaceidp1",
        ]
        assert set(get_workflow_keys(wm.workflow)) == create_key_set(files)


class TestMultipleCharactersNodesConnected:
    def test_one_char(self):
        wm = MultipleCharactersFaceIdWorkflowManager(n_characters=1)
        wm.trim_workflow()
        files = ["basic", "lora", "mcfaceid", "mcfaceidp0"]
        wm.save_workflow(
            os.path.join(output_dir, "multiple_characters", "mcfaceid_one_char.json")
        )
        assert set(get_workflow_keys(wm.workflow)) == create_key_set(files)

    def test_two_char(self):
        wm = MultipleCharactersFaceIdWorkflowManager(n_characters=2)
        wm.trim_workflow()
        files = ["basic", "lora", "mcfaceid", "mcfaceidp0", "mcfaceidp1"]
        wm.save_workflow(
            os.path.join(output_dir, "multiple_characters", "mcfaceid_two_char.json")
        )
        assert set(get_workflow_keys(wm.workflow)) == create_key_set(files)

    def test_three_char(self):
        wm = MultipleCharactersFaceIdWorkflowManager(n_characters=3)
        wm.trim_workflow()
        files = ["basic", "lora", "mcfaceid", "mcfaceidp0", "mcfaceidp1", "mcfaceidp2"]
        wm.save_workflow(
            os.path.join(output_dir, "multiple_characters", "mcfaceid_three_char.json")
        )
        assert set(get_workflow_keys(wm.workflow)) == create_key_set(files)
