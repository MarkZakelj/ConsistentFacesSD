import asyncio
import json
import os
import random
import uuid
from typing import Any

from comfy_requests import comfy_send_request
from PIL import Image
from prompt_construction import process_file
from tqdm import tqdm

from utils.imgs import image_exists, img_2_base64
from utils.paths import DATA_DIR, OUTPUT_DIR
from workflow_builder import TextToImageWorkflowManager

random.seed(42)

NSEEDS = 5

configs: dict[str, dict[str, Any]] = {
    "base_one_person": {
        "raw_prompts": "raw_prompts.txt",
        "person_codes": ["PERSON1"],
    },
    "ip_one_person": {
        "raw_prompts": "raw_prompts.txt",
        "person_codes": ["PERSON1"],
        "features": ["character"],
        "character": {"weight": 1.0},
    },
    "ip_one_person_wp5": {
        "raw_prompts": "raw_prompts.txt",
        "person_codes": ["PERSON1"],
        "features": ["character"],
        "character": {"weight": 0.5},
    },
    "ip_one_person_wp25": {
        "raw_prompts": "raw_prompts.txt",
        "person_codes": ["PERSON1"],
        "features": ["character"],
        "character": {"weight": 0.25},
    },
    "ip_one_person_wp5_sp35": {
        "raw_prompts": "raw_prompts.txt",
        "person_codes": ["PERSON1"],
        "features": ["character"],
        "character": {"weight": 0.5, "start_at": 0.35},
    },
    "ip_one_person_wp5_sp2": {
        "raw_prompts": "raw_prompts.txt",
        "person_codes": ["PERSON1"],
        "features": ["character"],
        "character": {"weight": 0.5, "start_at": 0.2},
    },
    "base_one_person_dreamshaper": {
        "raw_prompts": "raw_prompts.txt",
        "person_codes": ["PERSON1"],
        "checkpoint": "DreamShaperXL_Lightning.safetensors",
    },
    "base_two_people_dreamshaper": {
        "raw_prompts": "raw_prompts_two.txt",
        "person_codes": ["PERSON1", "PERSON2"],
        "checkpoint": "DreamShaperXL_Lightning.safetensors",
    },
    "base_two_people": {
        "raw_prompts": "raw_prompts_two.txt",
        "person_codes": ["PERSON1", "PERSON2"],
    },
    "base_three_people_dreamshaper": {
        "raw_prompts": "raw_prompts_two.txt",
        "person_codes": ["PERSON1", "PERSON2"],
        "checkpoint": "DreamShaperXL_Lightning.safetensors",
    },
}


async def generate_dataset(config_name: str):
    random.seed(42)
    conf = configs[config_name]
    raw_prompts_path = os.path.join(DATA_DIR, conf["raw_prompts"])
    lines = process_file(raw_prompts_path)

    img_save_path = os.path.join(OUTPUT_DIR, config_name)
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(os.path.join(img_save_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(img_save_path, "img_info"), exist_ok=True)
    total_images = len(lines) * NSEEDS
    info = {
        "n_images": total_images,
        "width": conf.get("width", 1024),
        "height": conf.get("height", 1024),
        "cfg": conf.get("cfg", 1.8),
        "steps": conf.get("steps", 6),
        "checkpoint": conf.get(
            "checkpoint", "juggernautXL_v9Rdphoto2Lightning.safetensors"
        ),
        "negative_prompt": conf.get(
            "negative_prompt",
            "weird, ugly, deformed, low contrast, bad anatomy, disfigured",
        ),
    } | conf
    print("INFO:")
    print(json.dumps(info, indent=2))
    json.dump(info, open(os.path.join(img_save_path, "info.json"), "w"), indent=2)

    pbar = tqdm(total=total_images, desc="Generating images")
    n_people = len(conf["person_codes"])
    prompt_seed_pairs = json.load(
        open(os.path.join(DATA_DIR, f"prompt_seed_pairs_{n_people}.json"), "r")
    )

    for i, line in enumerate(lines):
        for j in range(NSEEDS):
            img_num = i * NSEEDS + j
            prompt = prompt_seed_pairs[img_num]["prompt"]
            person_id_codes = list(prompt_seed_pairs[img_num]["people"].values())

            pbar.set_description(
                f"{config_name}: Processing image {img_num} - {prompt[:80]}..."
            )
            seed = prompt_seed_pairs[img_num]["seed"]

            if image_exists(config_name, img_num):
                pbar.update(1)
                continue

            img_info = {"prompt": prompt, "seed": seed}
            features = conf.get("features", [])

            wf = TextToImageWorkflowManager(features=features)

            wf.basic.set_prompt(prompt)
            wf.basic.set_seed(seed)
            wf.basic.set_negative_prompt(info["negative_prompt"])
            wf.basic.set_cfg(info["cfg"])
            wf.basic.set_width(info["width"])
            wf.basic.set_height(info["height"])
            wf.basic.set_steps(info["steps"])
            wf.basic.set_checkpoint(info["checkpoint"])

            profile_images = [
                Image.open(
                    os.path.join(
                        OUTPUT_DIR,
                        "identities",
                        "images_224x224",
                        f"{person_id_code}.png",
                    )
                )
                for person_id_code in person_id_codes
            ]

            if "character" in features:
                for p, profile_image in enumerate(profile_images):
                    img_info["character"] = {"code": p}
                    wf.character.set_image(img_2_base64(profile_image))
                    character = conf["character"]
                    wf.character.set_weight(character.get("weight", 1.0))
                    wf.character.set_end_at(character.get("end_at", 1.0))
                    wf.character.set_start_at(character.get("start_at", 0.0))
                    wf.character.set_weight_type(character.get("weight_type", "linear"))

            wf.trim_workflow()

            if img_num == 0:
                print(json.dumps(wf.get_workflow(), indent=2))

            req_id = str(uuid.uuid4())
            p = {"prompt": wf.get_workflow(), "client_id": req_id}
            prompt_seed_pairs.append(
                {
                    "prompt": prompt,
                    "seed": seed,
                    "people": dict(zip(conf["person_codes"], person_id_codes)),
                }
            )
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
                indent=2,
            )
            pbar.update(1)

    pbar.close()


async def main():
    print("Generating image datasets")
    for config_name in configs:
        print(f"Generating dataset for {config_name}")
        await generate_dataset(config_name)
        exit()


if __name__ == "__main__":
    asyncio.run(main())