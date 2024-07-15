import asyncio
import json
import os
import random
import uuid

from comfy_requests import comfy_send_request
from prompt_construction import process_file, replace_with_random_person
from tqdm import tqdm

from utils.imgs import image_exists
from utils.paths import DATA_DIR, OUTPUT_DIR
from workflow_builder import TextToImageWorkflowManager

random.seed(42)

NSEEDS = 5

configs: dict[str, dict[str, str]] = {
    "base_one_person": {
        "raw_prompts": "raw_prompts.txt",
        "person_codes": ["PERSON1"],
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
    }
    print("INFO:")
    print(json.dumps(info, indent=2))
    json.dump(info, open(os.path.join(img_save_path, "info.json"), "w"), indent=2)

    pbar = tqdm(total=total_images, desc="Generating images")

    for i, line in enumerate(lines):
        for j in range(NSEEDS):
            img_num = i * NSEEDS + j
            prompt = line
            for person_code in conf["person_codes"]:
                prompt = replace_with_random_person(prompt, person_code)
            pbar.set_description(
                f"{config_name}: Processing image {img_num} - {prompt[:80]}..."
            )
            seed = random.randint(0, 9999999999999)

            # make sure to execute all random calls first to keep randomness consistent
            if image_exists(config_name, img_num):
                pbar.update(1)
                continue

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
                indent=2,
            )

            pbar.update(1)

    pbar.close()


async def main():
    print("Generating image datasets")
    for config_name in configs:
        print(f"Generating dataset for {config_name}")
        await generate_dataset(config_name)


if __name__ == "__main__":
    asyncio.run(main())
