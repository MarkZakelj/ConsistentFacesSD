from workflow_builder import TextToImageWorkflowManager


def call_basic(wm):
    wm.basic.set_width(1234)
    wm.basic.set_height(1237)
    wm.basic.set_batch_size(3)
    wm.basic.set_prompt("((A storyboard rough pencil sketch)) of a man playing guitar")
    wm.basic.set_negative_prompt("blurry, low quality, cropped")
    wm.basic.set_checkpoint("checkpoint-latest.ckpt")
    wm.basic.set_seed(123456)
    wm.basic.set_steps(21)
    wm.basic.set_cfg(4.123)


def call_lora(wm):
    wm.lora.set_lora("lora-file-path", 1)
    wm.lora.set_strength_model(0.8, 1)
    wm.lora.set_lora("another-lora-file-path", 2)
    wm.lora.set_strength_model(0.5, 2)


def call_pose(wm):
    wm.pose.set_keypoints({"key": "value"})
    wm.pose.set_control_net_name("control-net-name.safetensors")
    wm.pose.set_strength(0.9)
    wm.pose.set_end_percent(50)


def call_face(wm):
    wm.face.set_steps(7)
    wm.face.set_cfg(1.34)


class TestWorkflowValueSetKeyExist:
    """
    tries to use the methods to set values in the workflow, to see, if the keys exist in the workflow
    """

    def test_basic(self):
        wm = TextToImageWorkflowManager()
        wm.trim_workflow()
        call_basic(wm)

    def test_lora(self):
        wm = TextToImageWorkflowManager(["lora"])
        wm.trim_workflow()
        call_basic(wm)
        call_lora(wm)

    def test_pose(self):
        wm = TextToImageWorkflowManager(["pose"])
        wm.trim_workflow()
        call_basic(wm)
        call_pose(wm)

    def test_face(self):
        wm = TextToImageWorkflowManager(["face"])
        wm.trim_workflow()
        call_basic(wm)
        call_face(wm)

    def test_all(self):
        wm = TextToImageWorkflowManager(["lora", "pose", "face"])
        wm.trim_workflow()
        call_basic(wm)
        call_lora(wm)
        call_pose(wm)
        call_face(wm)
