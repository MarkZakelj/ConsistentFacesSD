import asyncio
import json
import os
import random
import uuid

from loguru import logger

from comfy_requests import comfy_send_request
from prompt_construction import process_file, replace_with_random_person
from utils.paths import DATA_DIR, OUTPUT_DIR
from workflow_builder import TextToImageWorkflowManager

random.seed(42)

NSEEDS = 5


async def main():
    raw_prompts_path = os.path.join(DATA_DIR, "raw_prompts.txt")
    lines = process_file(raw_prompts_path)

    img_save_path = os.path.join(OUTPUT_DIR, "base_one_person")
    os.makedirs(img_save_path, exist_ok=True)

    info = {
        "n_images": len(lines) * NSEEDS,
        "width": 1024,
        "height": 1024,
        "cfg": 1.8,
        "steps": 6,
        "checkpoint": "juggernautXL_v9Rdphoto2Lightning.safetensors",
        "negative_prompt": "weird, ugly, deformed, low contrast, bad anatomy, disfigured",
    }
    json.dump(info, open(os.path.join(img_save_path, "info.json"), "w"), indent=2)
    for i, line in enumerate(lines):
        # use 3 different seeds for each prompt
        for j in range(NSEEDS):
            prompt = replace_with_random_person(line)
            img_num = i * NSEEDS + j
            logger.info(f"Processing image {img_num} - {prompt}")

            seed = random.randint(0, 9999999999999)
            img_info = {"prompt": prompt, "seed": seed}

            wf = TextToImageWorkflowManager()
            wf.basic.set_prompt(prompt)
            wf.basic.set_seed(seed)

            wf.basic.set_negative_prompt(info["negative_prompt"])
            wf.basic.set_cfg(info["cfg"])
            wf.basic.set_width(info["width"])
            wf.basic.set_height(info["height"])
            wf.basic.set_steps(info["steps"])
            wf.basic.set_checkpoint(info["checkpoint"])

            wf.trim_workflow()
            req_id = str(uuid.uuid4())
            p = {"prompt": wf.get_workflow(), "client_id": req_id}
            imgs, timings = await comfy_send_request(p, req_id)

            img = imgs[0]
            rgb_img = img.convert("RGB")
            img_num_string = f"{img_num:08}"
            rgb_img.save(
                os.path.join(img_save_path, "images", f"{img_num_string}.jpg"),
                "JPEG",
                quality=80,
            )
            json.dump(
                img_info,
                open(
                    os.path.join(img_save_path, "img_info", f"{img_num_string}.json"),
                    "w",
                ),
            )


if __name__ == "__main__":
    asyncio.run(main())
