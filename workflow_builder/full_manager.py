import json
import logging
from copy import deepcopy

from . import workflow_utils as wfu
from .manager_modules import (
    MAX_CHARACTERS,
    BasicManager,
    CharactersManager,
    CharactersManagerFaceID,
    CharactersManagerNormalIP,
    LoadImageManager,
    PoseManager,
    PoseMaskManager,
    SingleCharacterManager,
    SingleCharacterManagerFaceID,
)
from .nodes import FaceBoundingBox, FaceDetailer, ImageResize, LoraLoaderStack, Manager
from .workflow_utils import load_workflow, trim_workflow

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.DEBUG)


def check_valid_features(features, possible_features):
    for feature in features:
        if feature not in possible_features:
            raise ValueError(
                f"Feature {feature} is not available from {possible_features}"
            )


class FullManager(Manager):
    """Parent class for FullManagers.
    Full Managers have a dedicated workflow in json files and
    are used for setting the full workflow that is used in comfy
    """

    def workflow_to_string(self):
        return json.dumps(self.workflow, indent=4)

    def save_workflow(self, file_path="compiled_workflow.json"):
        json.dump(self.workflow, open(file_path, "w"), indent=4)

    def print_workflow(self):
        print(self.workflow_to_string())

    def get_workflow(self, trimmed=False, copy_wf=True):
        workflow = self.workflow
        if copy_wf:
            workflow = deepcopy(self.workflow)
        return workflow

    def trim_workflow(self):
        # use inline, to make the change for all submodules
        trim_workflow(self.workflow, inline=True)


class TextToImageWorkflowManager(FullManager):
    def __init__(self, features=None):
        super().__init__()
        possible_features = {"lora", "face", "pose", "character", "ip-faceid"}
        if features is None:
            features = []
        # check validity of features
        for feature in features:
            if feature not in possible_features:
                raise ValueError(
                    f"Feature {feature} is not available from {possible_features}"
                )

        # load the full workflow - later skip the unnecessary parts
        self.workflow = load_workflow("text_to_image_full.json")
        if "ip-faceid" in features:
            self.workflow = load_workflow("text_to_image_faceid_full.json")

        # initialize the composed managers (composition ftw)
        self.basic: BasicManager = BasicManager(workflow=self.workflow)
        self.lora: LoraLoaderStack = LoraLoaderStack(workflow=self.workflow)
        if "ip-faceid" in features:
            self.character: SingleCharacterManagerFaceID = SingleCharacterManagerFaceID(
                workflow=self.workflow, unique_name="Face"
            )
        else:
            self.character: SingleCharacterManager = SingleCharacterManager(
                workflow=self.workflow, unique_name="Face"
            )
        self.face: FaceDetailer = FaceDetailer(workflow=self.workflow)
        self.pose: PoseManager = PoseManager(workflow=self.workflow)

        # rewire workflow - skip the nodes, but not yet delete them (workflow is not trimmed at this point)
        if "lora" not in features:
            wfu.skip_node(
                "LoraLoaderStack(rgthree)0", [("model", 0), ("clip", 1)], self.workflow
            )

        if "character" not in features:
            wfu.skip_node("IPAdapterAdvancedFace0", [("model", 0)], self.workflow)
            wfu.skip_node("IPAdapterUnifiedLoader0", [("model", 0)], self.workflow)

        if "face" not in features:
            wfu.skip_node("FaceDetailer0", [("image", 0)], self.workflow)

        if "pose" not in features:
            if "ip-faceid" not in features:
                wfu.skip_node(
                    "ApplyControlNet(Advanced)0",
                    [("positive", 0), ("negative", 1)],
                    self.workflow,
                )

        if "nsfw_skip" in features:
            wfu.skip_node("NudenetDetector0", [("image", 0)], self.workflow)


class MultipleCharactersWorkflowManager(FullManager):
    def __init__(self, n_characters, features=None):
        super().__init__()
        if n_characters > MAX_CHARACTERS:
            n_characters = MAX_CHARACTERS
            # raise wf_errors.TooManyCharactersError(n_characters, MAX_CHARACTERS)
        possible_features = {"lora", "pose"}
        if features is None:
            features = []
        # check validity of features
        for feature in features:
            if feature not in possible_features:
                raise ValueError(
                    f"Feature {feature} is not available from {possible_features}"
                )

        # load the full workflow - later skip the unnecessary parts
        self.workflow = load_workflow("multiple_characters_full.json")

        # initialize the composed managers (composition ftw)
        self.basic: BasicManager = BasicManager(workflow=self.workflow)
        self.lora: LoraLoaderStack = LoraLoaderStack(workflow=self.workflow)
        self.pose: PoseManager = PoseManager(workflow=self.workflow)
        self.pose_mask: PoseMaskManager = PoseMaskManager(workflow=self.workflow)
        self.character: CharactersManager = CharactersManager(workflow=self.workflow)

        # rewire workflow - skip the nodes, but not yet delete them (workflow is not trimmed at this point)
        if "lora" not in features:
            wfu.skip_node(
                "LoraLoaderStack(rgthree)0", [("model", 0), ("clip", 1)], self.workflow
            )

        if "pose" not in features:
            wfu.skip_node(
                "ApplyControlNet(Advanced)PreGen0",
                [("positive", 0), ("negative", 1)],
                self.workflow,
            )

        if n_characters == 1:
            # no need to match characters, take just the biggest skeleton
            wfu.remove_inputs(
                "MaskFromPoints0", ["mask_mapping", "face_bbox"], self.workflow
            )

        self.pose_mask.set_n_poses(n_characters)
        # change this number to the max number of characters you want to use
        for i in range(n_characters, MAX_CHARACTERS):
            wfu.skip_node(f"IPAdapterAdvanced{i}0", [("model", 0)], self.workflow)
            wfu.skip_node(f"FaceDetailer{i}0", [("image", 0)], self.workflow)
            wfu.remove_inputs("MakeImageList0", [f"image{i}"], self.workflow)


class TwoCharactersFaceIdWorkflowManager(FullManager):
    def __init__(self):
        super().__init__()
        self.workflow = load_workflow("two_chars_faceid_full.json")

        self.basic: BasicManager = BasicManager(workflow=self.workflow)
        self.lora: LoraLoaderStack = LoraLoaderStack(workflow=self.workflow)
        self.pose_mask: PoseMaskManager = PoseMaskManager(workflow=self.workflow)
        self.character: CharactersManagerFaceID = CharactersManagerFaceID(
            workflow=self.workflow, ip_adapter_node_base_name="IPAdapterFaceID"
        )

        self.pose_mask.set_n_poses(2)


class MultipleCharactersFaceIdWorkflowManager(FullManager):
    def __init__(self, n_characters: int, features=None):
        super().__init__()
        if features is None:
            features = []

        check_valid_features(
            features,
            {"no-face-detailer", "no-facematch", "no-controlnet", "no-reference"},
        )
        # no-reference can be used when dealing with one subject as the IP-adapters wont have attention masks

        self.workflow = load_workflow("multiple_characters_workflow_full.json")

        self.basic: BasicManager = BasicManager(workflow=self.workflow)
        self.lora: LoraLoaderStack = LoraLoaderStack(workflow=self.workflow)
        self.pose_mask: PoseMaskManager = PoseMaskManager(workflow=self.workflow)
        self.character: CharactersManagerFaceID = CharactersManagerFaceID(
            workflow=self.workflow, ip_adapter_node_base_name="IPAdapterFaceID"
        )

        self.pose_mask.set_n_poses(10)
        for i in range(n_characters, MAX_CHARACTERS):
            wfu.skip_node(f"IPAdapterFaceID{i}0", [("model", 0)], self.workflow)
            wfu.skip_node(f"FaceDetailer{i}0", [("image", 0)], self.workflow)
            wfu.remove_inputs("MakeImageList0", [f"image{i+1}"], self.workflow)

        if "no-face-detailer" in features:
            for i in range(n_characters):
                wfu.skip_node(f"FaceDetailer{i}0", [("image", 0)], self.workflow)

        if "no-facematch" in features:
            wfu.remove_inputs("MaskFromPoints0", ["mask_mapping"], self.workflow)

        if "no-controlnet" in features:
            wfu.skip_node(
                "ApplyControlNet(Advanced)0",
                [("positive", 0), ("negative", 1)],
                self.workflow,
            )


class MultipleCharactersNormalIPWorkflowManager(FullManager):
    def __init__(self, n_characters: int, features=None):
        super().__init__()
        if features is None:
            features = []

        check_valid_features(
            features, {"no-face-detailer", "no-facematch", "no-controlnet"}
        )

        self.workflow = load_workflow(
            "multiple_characters_normal_ip_workflow_full.json"
        )

        self.basic: BasicManager = BasicManager(workflow=self.workflow)
        self.lora: LoraLoaderStack = LoraLoaderStack(workflow=self.workflow)
        self.pose_mask: PoseMaskManager = PoseMaskManager(workflow=self.workflow)
        self.character: CharactersManagerNormalIP = CharactersManagerNormalIP(
            workflow=self.workflow, ip_adapter_node_base_name="IPAdapter"
        )

        self.pose_mask.set_n_poses(10)
        for i in range(n_characters, MAX_CHARACTERS):
            wfu.skip_node(f"IPAdapter{i}0", [("model", 0)], self.workflow)
            wfu.skip_node(f"FaceDetailer{i}0", [("image", 0)], self.workflow)
            wfu.remove_inputs("MakeImageList0", [f"image{i+1}"], self.workflow)

        if "no-face-detailer" in features:
            for i in range(n_characters):
                wfu.skip_node(f"FaceDetailer{i}0", [("image", 0)], self.workflow)

        if "no-facematch" in features:
            wfu.remove_inputs("MaskFromPoints0", ["mask_mapping"], self.workflow)

        if "no-controlnet" in features:
            wfu.skip_node(
                "ApplyControlNet(Advanced)0",
                [("positive", 0), ("negative", 1)],
                self.workflow,
            )


class MCFaceIDFaceSwapWorkflowManager(FullManager):
    def __init__(self, n_characters, features=None):
        super().__init__()
        self.workflow = load_workflow("faceid_mc_full.json")

        possible_features = {"lora", "pose"}
        if features is None:
            features = []
        # check validity of features
        for feature in features:
            if feature not in possible_features:
                raise ValueError(
                    f"Feature {feature} is not available from {possible_features}"
                )

        self.basic: BasicManager = BasicManager(workflow=self.workflow)
        self.lora: LoraLoaderStack = LoraLoaderStack(workflow=self.workflow)
        self.pose: PoseManager = PoseManager(workflow=self.workflow)
        self.pose_mask: PoseMaskManager = PoseMaskManager(workflow=self.workflow)
        self.character: CharactersManagerFaceID = CharactersManagerFaceID(
            self.workflow, "IPAdapterFaceID"
        )

        if "pose" not in features:
            wfu.skip_node(
                "ApplyControlNet(Advanced)PreGen0",
                [("positive", 0), ("negative", 1)],
                self.workflow,
            )

        # if n_characters == 1:
        # no need to match characters, take just the biggest skeleton
        # wfu.remove_inputs(
        #     "MaskFromPoints0", ["mask_mapping", "face_bbox"], self.workflow
        # )
        # wfu.remove_inputs("ReActorFastFaceSwap00", ["face_bbox"], self.workflow)
        self.pose_mask.set_n_poses(n_characters)

        for i in range(n_characters, MAX_CHARACTERS):
            wfu.skip_node(f"IPAdapterFaceID{i}0", [("model", 0)], self.workflow)
            wfu.skip_node(
                f"ReActorFastFaceSwap{i}0", [("input_image", 0)], self.workflow
            )
            wfu.remove_inputs("MakeImageList0", [f"image{i+1}"], self.workflow)


class FaceStyleTransferWorkflowManager(FullManager):
    def __init__(self):
        super().__init__()
        self.workflow = load_workflow("face_style_transfer_full.json")

        self.basic: BasicManager = BasicManager(workflow=self.workflow)
        self.lora: LoraLoaderStack = LoraLoaderStack(workflow=self.workflow)
        self.image: LoadImageManager = LoadImageManager(workflow=self.workflow)


class FaceCropResizeWorkflowManager(FullManager):
    def __init__(self):
        super().__init__()
        self.workflow = load_workflow("face_crop_resize_full.json")

        self.image: LoadImageManager = LoadImageManager(workflow=self.workflow)
        self.resize: ImageResize = ImageResize(workflow=self.workflow)
        self.face_bbox: FaceBoundingBox = FaceBoundingBox(workflow=self.workflow)
