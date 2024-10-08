import os
import random

random.seed(42)

AGES = ["5-year-old", "18-year-old", "40-year-old", "70-year-old"]
CHILDREN = ["5-year-old"]
ADULTS = ["18-year-old", "40-year-old", "70-year-old"]
ETHNICITIES = [
    "black",
    "asian",
    "hispanic",
    "arab",
    "indian",
    "native american",
    "caucasian",
]
SEXES = ["man", "woman"]

ETH_MAP = {
    "bla": "black",
    "asi": "asian",
    "his": "hispanic",
    "ara": "arab",
    "ind": "indian",
    "nat": "native american",
    "cau": "caucasian",
}
SEX_MAP = {"man": "man", "wom": "woman"}
AGES_MAP = {
    "5": "5-year-old",
    "18": "18-year-old",
    "40": "40-year-old",
    "70": "70-year-old",
}

SEX_YOUNG = {"man": "boy", "woman": "girl"}


NUM_TO_WORD = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
}

prompt_enhancements = {
    "cinematic still": "(cinematic still) of {prompt} . high quality, detailed photograph, (visible face:0.8)",
}

NSEEDS = 5


def construct_mugshot_prompt(sex, age, ethnicity):
    if age == "5-year-old":
        sex = SEX_YOUNG[sex]
    return f"mugshot of a {age} {ethnicity} {sex}, detailed, high quality, full face photo, realistic, front facing, neutral background"


def construct_identity_prompt(sex, age, ethnicity):
    if age == "5-year-old":
        sex = SEX_YOUNG[sex]
    return f"face of a {age} {ethnicity} {sex}"


def construct_identity_prompt_from_code(person_code):
    age, ethnicity, sex = person_code.split("-")
    age = AGES_MAP[age]
    ethnicity = ETH_MAP[ethnicity]
    sex = SEX_MAP[sex]
    return construct_identity_prompt(sex, age, ethnicity)


def get_id_code(sex, age, ethnicity):
    age_code = age.split("-")[0]
    sex_code = sex[:3]
    ethnicity_code = ethnicity[:3]
    return f"{age_code}-{ethnicity_code}-{sex_code}"


def replace_with_random_child(line, person_code, sex=None):
    if sex is None:
        sex = random.choice(SEXES)
    age = random.choice(CHILDREN)
    ethnicity = random.choice(ETHNICITIES)
    person_id_code = get_id_code(sex, age, ethnicity)
    if person_code not in line:
        raise ValueError(f"Person code {person_code} not found in line")
    line = line.replace(person_code, f"{age} {ethnicity} {SEX_YOUNG[sex]}")
    return line, person_id_code


def replace_with_random_adult(line, person_code, sex=None):
    if sex is None:
        sex = random.choice(SEXES)
    age = random.choice(ADULTS)
    ethnicity = random.choice(ETHNICITIES)
    person_id_code = get_id_code(sex, age, ethnicity)
    if person_code not in line:
        raise ValueError(f"Person code {person_code} not found in line")
    line = line.replace(person_code, f"{age} {ethnicity} {sex}")
    return line, person_id_code


def replace_with_random_person(line, person_code):
    sex = random.choice(SEXES)
    age = random.choice(AGES)
    ethnicity = random.choice(ETHNICITIES)
    person_id_code = get_id_code(sex, age, ethnicity)
    if age == "5-year-old":
        sex = SEX_YOUNG[sex]
    if person_code not in line:
        raise ValueError(f"Person code {person_code} not found in line")
    line = line.replace(person_code, f"{age} {ethnicity} {sex}")
    return line, person_id_code


def enchance_prompt(prompt: str, style_name: str):
    pass


def process_file(file_path, n_people=1):
    processed_lines = []
    with open(file_path) as file:
        for line in file.readlines():
            if line.strip() == "":
                continue
            processed_line = (
                "(cinematic still) of "
                + line.strip()
                + " . high quality, detailed photograph, (visible face:0.8)"
            )
            if n_people > 1:
                processed_line = processed_line.replace(
                    ". high quality", f". {NUM_TO_WORD[n_people]} people, high quality"
                )
            # line_with_person_replaced = replace_person(processed_line)
            processed_lines.append(processed_line)
    return processed_lines


def main():
    file_path = os.path.join("data", "raw_prompts.txt")
    processed_lines = process_file(file_path)
    for line in processed_lines:
        print(line)


if __name__ == "__main__":
    main()
