import os

from utils.paths import TEST_DIR
from workflow_builder import TwoCharactersFaceIdWorkflowManager

from .test_workflow_set_value import call_basic


def call_character(wm: TwoCharactersFaceIdWorkflowManager, char_id: int):
    wm.character.set_image("image-base64", char_id)
    wm.character.set_weight(0.75, char_id)
    wm.character.set_start_at(10, char_id)


class TestWorkflowValueSetKeyExist:
    def test_basic(self):
        wm = TwoCharactersFaceIdWorkflowManager()
        wm.trim_workflow()
        call_basic(wm)
        call_character(wm, 0)
        call_character(wm, 1)
        wm.save_workflow(
            os.path.join(
                TEST_DIR, "test_workflow_builder/output/two_chars_faceid_full.json"
            )
        )
