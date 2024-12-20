"""This file contains the definitions of single nodes that are used inside comfy.
Using those nodes, we can model the exact workflow used inside comfy and enable programmatic setting of the values
"""


class Manager:
    """Base class for all managers, contains the workflow dictionary"""

    def __init__(self, workflow: dict = None):
        if workflow is None:
            workflow = {}
        self.workflow: dict = workflow

    def set_workflow(self, workflow):
        self.workflow = workflow


class FaceBoundingBox(Manager):
    def __init__(self, workflow, key="FaceBoundingBox0"):
        super().__init__(workflow)
        self.key = key

    def set_padding_percent(self, padding_percent: float):
        self.workflow[self.key]["inputs"]["padding_percent"] = padding_percent

    def set_force_square(self, force_square: bool):
        self.workflow[self.key]["inputs"]["force_square"] = force_square


class ImageResize(Manager):
    def __init__(self, workflow, key="ImageResize0"):
        super().__init__(workflow)
        self.key = key

    def set_resize_width(self, resize_width: int):
        self.workflow[self.key]["inputs"]["resize_width"] = resize_width

    def set_resize_height(self, resize_height: int):
        self.workflow[self.key]["inputs"]["resize_height"] = resize_height


class FaceDetailer(Manager):
    def __init__(self, workflow, key="FaceDetailer0"):
        super().__init__(workflow)
        self.key = key

    def set_steps(self, steps):
        self.workflow[self.key]["inputs"]["steps"] = steps

    def set_cfg(self, cfg):
        self.workflow[self.key]["inputs"]["cfg"] = cfg


class ApplyControlnet(Manager):
    def __init__(self, workflow, key="ApplyControlNet(Advanced)0"):
        super().__init__(workflow)
        self.key = key

    def set_strength(self, strength: float):
        self.workflow[self.key]["inputs"]["strength"] = strength

    def set_end_percent(self, end_percent):
        self.workflow[self.key]["inputs"]["end_percent"] = end_percent


class LoraLoaderStack(Manager):
    def __init__(self, workflow, keyname="LoraLoaderStack(rgthree)0"):
        super().__init__(workflow)
        self.keyname = keyname

    def set_lora(self, lora: str, lora_number: int):
        self.workflow[self.keyname]["inputs"][f"lora_0{lora_number}"] = lora

    def set_strength_model(self, strength: float, lora_number: int):
        self.workflow[self.keyname]["inputs"][f"strength_0{lora_number}"] = strength


class IPAdapter(Manager):
    def __init__(self, workflow: dict, keyname: str):
        super().__init__(workflow)
        self.keyname = keyname

    def set_weight(self, weight: float):
        self.workflow[self.keyname]["inputs"]["weight"] = weight

    def set_weight_type(self, weight_type: str):
        self.workflow[self.keyname]["inputs"]["weight_type"] = weight_type

    def set_start_at(self, start_at):
        self.workflow[self.keyname]["inputs"]["start_at"] = start_at

    def set_end_at(self, end_at):
        self.workflow[self.keyname]["inputs"]["end_at"] = end_at


class IPAdapterFaceID(IPAdapter):
    def __init__(self, workflow, keyname):
        super().__init__(workflow, keyname)

    def set_weight_v2(self, weight: float):
        self.workflow[self.keyname]["inputs"]["weight_faceidv2"] = weight


class FaceMatcher(Manager):
    def __init__(self, workflow, keyname="FaceMatcher0"):
        super().__init__(workflow)
        self.keyname = keyname

    def set_reverse(self, reverse: bool):
        self.workflow[self.keyname]["inputs"]["reverse"] = reverse
