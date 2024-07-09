import json
import uuid
from random import randint

import pytest
from image_creation.comfy_requests import comfy_send_request

from tests import save_img_from_comfy, save_timings, save_workflow
from workflow_builder import TextToImageWorkflowManager


def save_all(imgs, timings, workflow, test_name):
    save_img_from_comfy(imgs, test_name)
    save_timings(timings, test_name)
    save_workflow(workflow, test_name)


@pytest.mark.asyncio
async def test_connection():
    wf = TextToImageWorkflowManager()
    wf.basic.set_prompt(
        "(cinematic still) of a young man enjoying the sunset on the beach, with a surfboard, high quality"
    )
    wf.basic.set_negative_prompt("weird, ugly, deformed, low contrast, bad")
    wf.basic.set_height(1024)
    wf.basic.set_width(1024)
    seed = randint(0, 9999999999)
    wf.basic.set_seed(seed)
    wf.trim_workflow()
    wf_trimmed = wf.get_workflow()
    req_id = str(uuid.uuid4())
    p = {"prompt": wf_trimmed, "client_id": req_id}
    print(json.dumps(wf_trimmed, indent=4))
    imgs, timings = await comfy_send_request(p, req_id)
    assert len(imgs) == 1
    save_all(imgs, timings, wf_trimmed, "test_connection")
