import json
import os

from workflow_builder import workflow_utils as wfu

script_dir = os.path.dirname(os.path.realpath(__file__))

output_dir = os.path.join(script_dir, "output")

os.makedirs(os.path.join(output_dir, "text_to_image"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "multiple_characters"), exist_ok=True)


class TestWorkflowUtils:
    # def test_convert_load64_to_load(self):
    #     workflow = wfu.load_workflow("text_to_image_full.json")
    #     workflow = wfu.convert_loadb64_to_load(workflow)
    #     for node in workflow.keys():
    #         if node.startswith("LoadImage(Base64)"):
    #             raise AssertionError(f"LoadImage(Base64) node found in workflow: {node}")
    #         if node.startswith("LoadMask(Base64)"):
    #             raise AssertionError(f"LoadMask(Base64) node found in workflow: {node}")
    #     json.dump(workflow, open(os.path.join(output_dir, "text_to_image_full_loadimage.json"), "w"), indent=4)

    def test_build_digraph(self):
        workflow = wfu.load_workflow("text_to_image_full.json")
        digraph = wfu.build_digraph(workflow)
        json.dump(
            digraph,
            open(os.path.join(output_dir, "text_to_image_full_digraph.json"), "w"),
            indent=4,
        )
