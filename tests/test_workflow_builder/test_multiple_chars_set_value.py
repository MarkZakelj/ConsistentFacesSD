from workflow_builder import MultipleCharactersWorkflowManager
from workflow_builder.errors import CharacterIdNotInRangeError

from .test_workflow_set_value import call_basic


def call_lora(wm):
    wm.lora.set_lora("lora-file-path", 1)
    wm.lora.set_strength_model(0.8, 1)
    wm.lora.set_lora("another-lora-file-path", 2)
    wm.lora.set_strength_model(0.5, 2)


def call_character(wm, char_id: int):
    wm.character.set_image("image-base64", char_id)
    wm.character.set_weight(0.75, char_id)
    wm.character.set_noise(0.1, char_id)
    wm.character.set_start_at(10, char_id)


def call_pose(wm):
    wm.pose_mask.set_keypoints({"key": "value"})
    wm.pose_mask.set_use_keypoints("face+shoulders")
    wm.pose_mask.set_dilate_iterations(3)
    wm.pose_mask.set_prompt("A man playing guitar")
    wm.pose_mask.set_steps(21)
    wm.pose_mask.set_seed(123456)


class TestWorkflowValueSetKeyExist:
    """
    tries to use the methods to set values in the workflow, to see, if the keys exist in the workflow
    """

    def test_basic(self):
        wm = MultipleCharactersWorkflowManager(2, None)
        wm.trim_workflow()
        call_basic(wm)
        call_character(wm, 0)
        call_character(wm, 1)

    def test_lora(self):
        wm = MultipleCharactersWorkflowManager(2, ["lora"])
        wm.trim_workflow()
        call_lora(wm)
        call_basic(wm)

    def test_one_character(self):
        wm = MultipleCharactersWorkflowManager(1, None)
        wm.trim_workflow()
        call_basic(wm)
        call_character(wm, 0)

    def test_error_on_too_much_characters(self):
        wm = MultipleCharactersWorkflowManager(2, None)
        wm.trim_workflow()
        call_basic(wm)
        call_character(wm, 0)
        call_character(wm, 1)
        try:
            call_character(wm, 99999999999)
        except CharacterIdNotInRangeError:
            pass

    def test_pose_mask_node(self):
        wm = MultipleCharactersWorkflowManager(2, ["pose"])
        wm.trim_workflow()
        call_basic(wm)
        call_character(wm, 0)
        call_character(wm, 1)
        call_pose(wm)


class TestImageTypeSwitcher:
    def test_load_node_type_conversion(self):
        wm = MultipleCharactersWorkflowManager(2, ["lora"])
        wm.trim_workflow()
        wm.character.set_image("imagebase64codeproxy", 0)
        wm.character.set_image("18-asi-mal.png", 1)
        assert "LoadImage(Base64)IP00" in wm.workflow
        assert "LoadImageIP10" in wm.workflow
        assert (
            "imagebase64codeproxy"
            == wm.workflow["LoadImage(Base64)IP00"]["inputs"]["image"]
        )
        assert "18-asi-mal.png" == wm.workflow["LoadImageIP10"]["inputs"]["image"]
