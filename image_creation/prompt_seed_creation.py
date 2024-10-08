import json
import os

from image_creation.prompt_construction import (
    process_file,
    replace_with_random_adult,
    replace_with_random_child,
    replace_with_random_person,
)
from utils import get_random_comfy_seed
from utils.paths import DATA_DIR

PERSON_CODES = [
    "PERSON1",
    "PERSON2",
    "PERSON3",
    "PERSON4",
    "PERSON5",
]

NSEEDS = 5


def create_from_raw_prompts(n_people: int):
    filepath = os.path.join(DATA_DIR, f"raw_prompts_{n_people}.txt")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found")
    lines = process_file(filepath, n_people=n_people)
    ps_pairs = []
    for i, line in enumerate(lines):
        for s in range(NSEEDS):
            prompt = line
            people = {}
            for n in range(n_people):
                prompt, person_id_code = replace_with_random_person(
                    prompt, PERSON_CODES[n]
                )
                people[PERSON_CODES[n]] = person_id_code
            seed = get_random_comfy_seed()
            pair = {
                "prompt": prompt,
                "seed": seed,
                "people": people,
                "img_code": i * NSEEDS + s,
            }
            ps_pairs.append(pair)
    return ps_pairs


def create_child_adult_prompt_seeds():
    # use for 2 people
    filepath = os.path.join(DATA_DIR, "raw_prompts_child_adult_2.txt")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found")
    lines = process_file(filepath, n_people=2)
    ps_pairs = []
    for i, line in enumerate(lines):
        for s in range(NSEEDS):
            prompt = line
            people = {}
            prompt, person_id_code = replace_with_random_child(prompt, PERSON_CODES[0])
            people[PERSON_CODES[0]] = person_id_code
            prompt, person_id_code = replace_with_random_adult(prompt, PERSON_CODES[1])
            people[PERSON_CODES[1]] = person_id_code
            seed = get_random_comfy_seed()
            pair = {
                "prompt": prompt,
                "seed": seed,
                "people": people,
                "img_code": i * NSEEDS + s,
            }
            ps_pairs.append(pair)
    return ps_pairs


def copy_and_change_seeds(n_people: int):
    filepath = os.path.join(DATA_DIR, f"prompt_seed_pairs_{n_people}.json")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found")
    with open(filepath, "r") as file:
        pairs = json.load(file)
    new_pairs = []
    for pair in pairs:
        new_pair = pair.copy()
        new_pair["seed"] = get_random_comfy_seed()
        new_pairs.append(new_pair)
    new_filepath = os.path.join(DATA_DIR, f"prompt_seed_pairs_{n_people}_new.json")
    with open(new_filepath, "w") as file:
        json.dump(new_pairs, file, indent=2)


COPY = False
CHILD_ADULT = True
N_PEOPLE = [2]


def main():
    for n in N_PEOPLE:
        if COPY:
            copy_and_change_seeds(n)
            continue
        if CHILD_ADULT:
            filepath = os.path.join(DATA_DIR, f"prompt_seed_pairs_child_adult_{n}.json")
            if os.path.exists(filepath):
                continue
            pairs = create_child_adult_prompt_seeds()
            print("LEN pairs", len(pairs))
            json.dump(pairs, open(filepath, "w"), indent=2)
            continue
        filepath = os.path.join(DATA_DIR, f"prompt_seed_pairs_{n}.json")
        # if os.path.exists(filepath):
        #     continue
        pairs = create_from_raw_prompts(n)
        print("LEN pairs", len(pairs))
        with open(filepath, "w") as file:
            json.dump(pairs, file, indent=2)


if __name__ == "__main__":
    main()
