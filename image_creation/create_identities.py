import asyncio
import json
import os
import random
from itertools import product

from prompt_construction import AGES, ETHNICITIES, SEXES
from tqdm import tqdm

from image_creation.comfy_requests import send_workflow_to_comfy
from image_creation.prompt_construction import construct_mugshot_prompt, get_id_code
from utils.imgs import image_exists
from utils.paths import OUTPUT_DIR
from workflow_builder import TextToImageWorkflowManager

random.seed(42)


async def main():
    subset_name = "identities"
    img_save_path = os.path.join(OUTPUT_DIR, subset_name)
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(os.path.join(img_save_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(img_save_path, "img_info"), exist_ok=True)

    info = dict(
        width=1024,
        height=1024,
        batch_size=1,
        negative_prompt="weird, ugly, deformed, shade",
        checkpoint="juggernautXL_v9Rdphoto2Lightning.safetensor",
        steps=6,
        cfg=1.5,
    )

    json.dump(
        info,
        open(os.path.join(img_save_path, "info.json"), "w"),
        indent=2,
    )
    pbar = tqdm(
        total=len(SEXES) * len(AGES) * len(ETHNICITIES), desc="Generating images"
    )
    for sex, age, ethnicity in product(SEXES, AGES, ETHNICITIES):
        img_filename = get_id_code(sex, age, ethnicity)
        seed = random.randint(0, 99999999999)

        pbar.set_description(f"Generating img: {img_filename}")
        if image_exists(subset_name, img_filename):
            pbar.update(1)
            continue

        prompt = construct_mugshot_prompt(sex, age, ethnicity)
        wf_manager = TextToImageWorkflowManager()
        wf_manager.basic.set_all(
            width=1024,
            height=1024,
            batch_size=1,
            prompt=prompt,
            negative_prompt="weird, ugly, deformed, shade",
            checkpoint="juggernautXL_v9Rdphoto2Lightning.safetensors",
            seed=seed,
            steps=6,
            cfg=1.5,
        )
        img_info = dict(prompt=prompt, seed=seed)

        wf_manager.trim_workflow()
        imgs = await send_workflow_to_comfy(wf_manager)
        img = imgs[0]
        img = img.convert("RGB")

        img.save(
            os.path.join(img_save_path, "images", f"{img_filename}.jpg"),
            "JPEG",
            quality=80,
        )
        json.dump(
            img_info,
            open(os.path.join(img_save_path, "img_info", f"{img_filename}.json"), "w"),
            indent=2,
        )
        pbar.update(1)


if __name__ == "__main__":
    asyncio.run(main())
