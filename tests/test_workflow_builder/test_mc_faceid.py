from workflow_builder import MCFaceIDFaceSwapWorkflowManager

from .test_multiple_chars_set_value import call_basic, call_lora


def call_pose(wf):
    wf.pose.set_keypoints({"key": "value"})
    wf.pose.set_control_net_name("control-net-name.safetensors")
    wf.pose.set_strength(0.9)
    wf.pose.set_end_percent(50)


def call_character_faceid(wf, character_id):
    wf.character.set_image("image-base64", character_id)
    wf.character.set_weight(0.75, character_id)
    wf.character.set_start_at(0.3, character_id)
    wf.character.set_weight_type("linear", character_id)


class TestMCFaceIDSetValue:
    def test_basic(self):
        wf = MCFaceIDFaceSwapWorkflowManager(1)
        call_basic(wf)

    def test_lora(self):
        wf = MCFaceIDFaceSwapWorkflowManager(1, ["lora"])
        call_basic(wf)
        call_lora(wf)

    def test_pose(self):
        wf = MCFaceIDFaceSwapWorkflowManager(1, ["pose"])
        call_basic(wf)
        call_pose(wf)

    def test_character(self):
        wf = MCFaceIDFaceSwapWorkflowManager(1)
        call_basic(wf)
        call_character_faceid(wf, 0)

    def test_two_characters(self):
        wf = MCFaceIDFaceSwapWorkflowManager(2)
        call_basic(wf)
        call_character_faceid(wf, 0)
        call_character_faceid(wf, 1)

    def test_feature_not_exist(self):
        try:
            MCFaceIDFaceSwapWorkflowManager(1, ["-----"])
        except ValueError:
            pass
        else:
            assert False, "Should raise ValueError"
