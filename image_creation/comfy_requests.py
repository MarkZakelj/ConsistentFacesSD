import asyncio
import json
import os
import uuid
from io import BytesIO

import requests
import websockets
from dotenv import load_dotenv
from loguru import logger
from PIL import Image

from utils import SequentialTimer, get_exception_traceback_str
from workflow_builder.full_manager import FullManager
from workflow_builder.workflow_utils import renumber_workflow

load_dotenv()

JSONHEADERS = {"Content-Type": "application/json"}

COMFY_HOST = os.getenv("COMFY_HOST", "localhost")
COMFY_PORT = os.getenv("COMFY_PORT", "")
logger.info("COMFY HOST: " + COMFY_HOST)
logger.info("COMFY PORT: " + COMFY_PORT)

ws_uri_template = f"ws://{COMFY_HOST}:{COMFY_PORT}/ws?clientId={{client_id}}"
comfy_url = f"http://{COMFY_HOST}:{COMFY_PORT}/prompt"

# if COMFY_HOST.startswith('http'):
#     comfy_url = f"{COMFY_HOST}/prompt"
#     ws_uri_template = f"{COMFY_HOST}/ws?clientId={{client_id}}"


async def websocket_get_image(prompt_id: str, client_id: str, verbose=False):
    imgs = []
    uri = ws_uri_template.format(client_id=client_id)
    async with websockets.connect(
        uri, max_size=2**25, close_timeout=200, open_timeout=160, ping_timeout=100
    ) as ws:
        st = SequentialTimer()
        while True:
            # error is thrown if timeout - happens when comfy receives identical payload
            # and caches even the websocket send image node-nothing is sent
            out = await asyncio.wait_for(ws.recv(), timeout=100)

            if isinstance(out, str):
                message = json.loads(out)
                if message["type"] == "executing":
                    data = message["data"]
                    if data["node"] is None and data["prompt_id"] == prompt_id:
                        st.time("END")
                        break  # Execution is done
                if "node" in message["data"]:
                    st.time(message["data"]["node"])
                    if verbose:
                        print(message["data"]["node"])
                else:
                    st.time(str(message))
            elif isinstance(out, bytes):
                bts = BytesIO(out[8:])  # remove header from payload
                img = Image.open(bts)
                imgs.append(img)
                if verbose:
                    logger.info("IMAGE RECEIVED")
                continue
    if verbose:
        print("FINISHED")
    return imgs, st.get_json_merged()


async def comfy_send_request(payload, req_id):
    imgs = []
    timings = {}
    if "prompt" not in payload:
        raise ValueError("Payload does not contain prompt")
    payload["prompt"] = renumber_workflow(payload["prompt"])
    try:
        response = requests.post(comfy_url, json=payload, headers=JSONHEADERS)
        if response.status_code != 200:
            logger.error(
                "RequestFailed with non 200 status code, get new address\nSTATUS CODE: {status_code}\nERROR: {error}",
                status_code=response.status_code,
                error=response.text,
            )
        else:
            res = response.json()
            prompt_id = res["prompt_id"]
            imgs, timings = await websocket_get_image(prompt_id, req_id, verbose=False)
            if len(imgs) == 0:
                logger.error(
                    "No images received from comfy, TIMINGS:\n{timings}",
                    timings=timings,
                )
    except Exception as e:
        logger.error(
            "RequestFailed, ERROR:\n{error}",
            error=str(get_exception_traceback_str(e)),
        )
    return imgs, timings


async def send_workflow_to_comfy(workflow: FullManager):
    req_id = str(uuid.uuid4())
    p = {"prompt": workflow.get_workflow(), "client_id": req_id}
    imgs, timings = await comfy_send_request(p, req_id)
    return imgs
