import inspect
import json
from functools import wraps
from typing import Literal

from workflow_builder import workflow_utils as wfu
from workflow_builder.errors import CharacterIdNotInRangeError
from workflow_builder.nodes import ApplyControlnet, IPAdapter, IPAdapterFaceID, Manager
from workflow_builder.workflow_utils import is_image_filename

MAX_CHARACTERS = 3  # maximum number of characters that can be used in the multiple character workflow


class BasicManager(Manager):
    def __init__(self, workflow=None):
        super().__init__(workflow)

    def set_width(self, width: int):
        self.workflow["Width0"]["inputs"]["number"] = width

    def set_height(self, height: int):
        self.workflow["Height0"]["inputs"]["number"] = height

    def set_batch_size(self, batch_size: int):
        self.workflow["EmptyLatentImage0"]["inputs"]["batch_size"] = batch_size

    def set_prompt(self, prompt: str):
        self.workflow["CLIPTextEncode(Prompt)Positive0"]["inputs"]["text"] = prompt

    def set_negative_prompt(self, negative_prompt: str):
        self.workflow["CLIPTextEncode(Prompt)Negative0"]["inputs"][
            "text"
        ] = negative_prompt

    def set_checkpoint(self, checkpoint: str):
        self.workflow["LoadCheckpoint0"]["inputs"]["ckpt_name"] = checkpoint

    def set_seed(self, seed: int):
        self.workflow["Seed(rgthree)0"]["inputs"]["seed"] = seed

    def set_steps(self, steps: int):
        self.workflow["KSampler0"]["inputs"]["steps"] = steps

    def set_cfg(self, cfg: float):
        self.workflow["KSampler0"]["inputs"]["cfg"] = cfg

    def set_all(
        self,
        width: int,
        height: int,
        batch_size: int,
        prompt: str,
        negative_prompt: str,
        checkpoint: str,
        seed: int,
        steps: int,
        cfg: float,
    ):
        self.set_width(width)
        self.set_height(height)
        self.set_batch_size(batch_size)
        self.set_prompt(prompt)
        self.set_negative_prompt(negative_prompt)
        self.set_checkpoint(checkpoint)
        self.set_seed(seed)
        self.set_steps(steps)
        self.set_cfg(cfg)


LoadImageNodeType = Literal["base64", "normal"]


class ImageLoadSwitcher(Manager):
    """used to change the LoadImage node type to LoadImageBase64 or do the reverse
    Should be used as a composable element inside other managers
    LoadImage node requires an actual image to be present on the server
            (can be included in docker build or uploaded via comfyUI beforehand)
    LoadImageBase64 node requires the image to be base64 encoded and present in the workflow request
    """

    def __init__(self, workflow: dict, base64_node: str, normal_node: str):
        super().__init__(workflow)
        self.base64_node = base64_node
        self.normal_node = normal_node
        self.load_image_node = None
        if base64_node in workflow:
            self.load_image_node = base64_node
        elif normal_node in workflow:
            self.load_image_node = normal_node
        else:
            keys = sorted([k for k in workflow])
            for key in keys:
                print(key)
            raise ValueError(
                f"Neither {base64_node} nor {normal_node} found in the workflow"
            )

    def set_load_image_type(
        self,
        node_type: LoadImageNodeType,
    ):
        """Whether to use Normal LoadImage node or LoadImageBase64 node
        Args:
            node_type: 'base64' or 'normal'
        """
        if node_type == "base64":
            if self.load_image_node == self.base64_node:
                return
            wfu.convert_load_to_loadb64(
                self.workflow, target=self.normal_node, inline=True
            )
            wfu.replace_key(
                self.normal_node, self.base64_node, self.workflow, inline=True
            )
            self.load_image_node = self.base64_node
        else:
            if self.load_image_node == self.normal_node:
                return
            wfu.convert_loadb64_to_load(
                self.workflow, target=self.base64_node, inline=True
            )
            wfu.replace_key(
                self.base64_node, self.normal_node, self.workflow, inline=True
            )
            self.load_image_node = self.normal_node

    def set_load_image_type_from_string(self, image_str: str):
        """Set the LoadImage type based on the image string
        Args:
            image_str: string that contains the base64 image or name of the image with extension e.g. 18-asi-mal.png
        """
        if is_image_filename(image_str):
            self.set_load_image_type("normal")
        else:
            self.set_load_image_type("base64")


class LoadImageManager(Manager):
    def __init__(self, workflow: dict, unique_name="", number=0):
        super().__init__(workflow)
        self.image_load_switcher = ImageLoadSwitcher(
            workflow,
            f"LoadImage(Base64){unique_name}{number}",
            f"LoadImage{unique_name}{number}",
        )

    def set_image(self, image: str):
        self.image_load_switcher.set_load_image_type_from_string(image)
        self.workflow[self.image_load_switcher.load_image_node]["inputs"][
            "image"
        ] = image


def check_character_id(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Get the signature of the function
        sig = inspect.signature(func)
        # Bind the passed arguments to the signature
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()

        character_id = bound_args.arguments.get("character_id")

        if character_id is None:
            raise ValueError("character_id must be provided")
        if character_id < 0 or character_id >= MAX_CHARACTERS:
            raise CharacterIdNotInRangeError(character_id, MAX_CHARACTERS - 1)
        # only execute the method under this wrapper, if the character_id is valid (small enough)
        return func(self, *args, **kwargs)

    return wrapper


class SingleCharacterManager(Manager):
    def __init__(self, workflow: dict, unique_name: str = ""):
        super().__init__(workflow)

        self.load_image = LoadImageManager(workflow, unique_name)
        self.ip_adapter = IPAdapter(workflow, f"IPAdapterAdvanced{unique_name}0")

    def set_image(self, *args, **kwargs):
        self.load_image.set_image(*args, **kwargs)

    def set_weight(self, *args, **kwargs):
        self.ip_adapter.set_weight(*args, **kwargs)

    def set_weight_type(self, *args, **kwargs):
        self.ip_adapter.set_weight_type(*args, **kwargs)

    def set_start_at(self, *args, **kwargs):
        self.ip_adapter.set_start_at(*args, **kwargs)

    def set_end_at(self, *args, **kwargs):
        self.ip_adapter.set_end_at(*args, **kwargs)


class SingleCharacterManagerFaceID(SingleCharacterManager):
    def __init__(self, workflow: dict, unique_name: str = ""):
        super().__init__(workflow, unique_name)
        self.ip_adapter = IPAdapterFaceID(workflow, f"IPAdapterFaceID{unique_name}0")

    def set_weight_v2(self, *args, **kwargs):
        self.ip_adapter.set_weight_v2(*args, **kwargs)


class CharactersManager(Manager):
    def __init__(self, workflow: dict):
        super().__init__(workflow)
        # allow up to 5 characters for now
        self.image_load_switchers = [
            ImageLoadSwitcher(workflow, f"LoadImage(Base64)IP{i}0", f"LoadImageIP{i}0")
            for i in range(MAX_CHARACTERS)
        ]

    @check_character_id
    def set_image(self, image: str, character_id: int):
        """Set the image for a specific character
        Args:
            image: either base64 string of an image or image name e.g. 18-asi-mal.png
            character_id: number of the character, 0-indexed
        """
        self.image_load_switchers[character_id].set_load_image_type_from_string(image)
        self.workflow[self.image_load_switchers[character_id].load_image_node][
            "inputs"
        ]["image"] = image

    @check_character_id
    def set_weight(self, weight: float, character_id: int):
        self.workflow[f"IPAdapterAdvanced{character_id}0"]["inputs"]["weight"] = weight

    @check_character_id
    def set_weight_type(self, weight_type: str, character_id: int):
        self.workflow[f"IPAdapterAdvanced{character_id}0"]["inputs"][
            "weight_type"
        ] = weight_type

    @check_character_id
    def set_noise(self, noise, character_id: int):
        self.workflow[f"IPAdapterAdvanced{character_id}0"]["inputs"]["noise"] = noise

    @check_character_id
    def set_start_at(self, start_at, character_id: int):
        self.workflow[f"IPAdapterAdvanced{character_id}0"]["inputs"][
            "start_at"
        ] = start_at


class CharactersManagerNormalIP(Manager):
    def __init__(self, workflow: dict, ip_adapter_node_base_name: str):
        """Args:
        ----
            workflow: the workflow dict
            ip_adapter_node_base_name: node key in the workflow that manages the normal IPadapter  for character
            load_image_special_name: whatever is added after 'LoadImage' in the Load Image node key, for example

        """
        super().__init__(workflow)
        self.image_loaders = [
            LoadImageManager(workflow, unique_name=f"Face{i}")
            for i in range(MAX_CHARACTERS)
        ]
        self.ip_adapters = [
            IPAdapter(workflow, keyname=f"{ip_adapter_node_base_name}{i}0")
            for i in range(MAX_CHARACTERS)
        ]

    @check_character_id
    def set_image(self, image: str, character_id: int):
        """Set the image for a specific character
        Args:
            image: either base64 string of an image or image name e.g. 18-asi-mal.png
            character_id: number of the character, 0-indexed
        """
        self.image_loaders[character_id].set_image(image)

    @check_character_id
    def set_weight(self, weight: float, character_id: int):
        self.ip_adapters[character_id].set_weight(weight)

    @check_character_id
    def set_weight_type(self, weight_type: str, character_id: int):
        self.ip_adapters[character_id].set_weight_type(weight_type)

    @check_character_id
    def set_start_at(self, start_at, character_id: int):
        self.ip_adapters[character_id].set_start_at(start_at)


class CharactersManagerFaceID(Manager):
    def __init__(self, workflow: dict, ip_adapter_node_base_name: str):
        """Args:
        ----
            workflow: the workflow dict
            ip_adapter_node_base_name: node key in the workflow that manages the IPadapter for character
            load_image_special_name: whatever is added after 'LoadImage' in the Load Image node key, for example

        """
        super().__init__(workflow)
        self.image_loaders = [
            LoadImageManager(workflow, unique_name=f"Face{i}")
            for i in range(MAX_CHARACTERS)
        ]
        self.ip_adapters = [
            IPAdapterFaceID(workflow, keyname=f"{ip_adapter_node_base_name}{i}0")
            for i in range(MAX_CHARACTERS)
        ]

    @check_character_id
    def set_image(self, image: str, character_id: int):
        """Set the image for a specific character
        Args:
            image: either base64 string of an image or image name e.g. 18-asi-mal.png
            character_id: number of the character, 0-indexed
        """
        self.image_loaders[character_id].set_image(image)

    @check_character_id
    def set_weight(self, weight: float, character_id: int):
        self.ip_adapters[character_id].set_weight(weight)

    @check_character_id
    def set_weight_type(self, weight_type: str, character_id: int):
        self.ip_adapters[character_id].set_weight_type(weight_type)

    @check_character_id
    def set_start_at(self, start_at, character_id: int):
        self.ip_adapters[character_id].set_start_at(start_at)

    @check_character_id
    def set_weight_v2(self, weight_v2: float, character_id: int):
        self.ip_adapters[character_id].set_weight_v2(weight_v2)


class PoseManager(Manager):
    def __init__(self, workflow):
        super().__init__(workflow)
        self.apply_controlnet = ApplyControlnet(
            workflow, key="ApplyControlNet(Advanced)0"
        )

    def set_keypoints(self, pose_keypoint: str):
        self.workflow["LoadPosesJSON0"]["inputs"]["pose_keypoint"] = pose_keypoint

    def set_control_net_name(self, control_net_name: str):
        self.workflow["LoadControlNetModel0"]["inputs"][
            "control_net_name"
        ] = control_net_name

    def set_strength(self, strength: float):
        self.apply_controlnet.set_strength(strength)

    def set_end_percent(self, end_percent):
        self.apply_controlnet.set_end_percent(end_percent)


class PoseMaskManager(Manager):
    def __init__(self, workflow):
        super().__init__(workflow)
        self.apply_pre_controlnet = ApplyControlnet(
            workflow, key="ApplyControlNet(Advanced)PreGen0"
        )
        self.apply_main_controlnet = ApplyControlnet(
            workflow, key="ApplyControlNet(Advanced)0"
        )

    def set_strength(self, strength: float):
        self.apply_pre_controlnet.set_strength(strength)
        # self.apply_main_controlnet.set_strength(strength)

    def set_end_percent(self, end_percent: float):
        self.apply_pre_controlnet.set_end_percent(end_percent)
        # self.apply_main_controlnet.set_end_percent(end_percent)

    def set_keypoints(self, keypoints: dict):
        self.workflow["LoadPosesJSON0"]["inputs"]["pose_keypoint"] = json.dumps(
            keypoints
        )

    def set_use_keypoints(self, use_keypoints: str):
        """Set the used keypoints for the MaskFromPoints node - which points from skeleton to use for mask creation
        Args:
            use_keypoints: 'face' | 'face+shoulders' | 'face+torso' | 'full-body
        """
        self.workflow["MaskFromPoints0"]["inputs"]["use_keypoints"] = use_keypoints

    def set_n_poses(self, n_poses: int):
        self.workflow["MaskFromPoints0"]["inputs"]["n_poses"] = n_poses

    def set_dilate_iterations(self, dilate_iterations: int):
        self.workflow["MaskFromPoints0"]["inputs"][
            "dilate_iterations"
        ] = dilate_iterations

    def set_prompt(self, prompt: str):
        self.workflow["CLIPTextEncode(Prompt)PositivePose0"]["inputs"]["text"] = prompt

    def set_steps(self, steps: int):
        self.workflow["KSamplerPose0"]["inputs"]["steps"] = steps

    def set_seed(self, seed: int):
        self.workflow["KSamplerPose0"]["inputs"]["seed"] = seed

    def set_reverse_facematch(self, reverse_facematch: bool):
        self.workflow["KSamplerPose0"]["inputs"][
            "reverse_facematch"
        ] = reverse_facematch
