import os
import random

random.seed(42)

AGES = ["5-year-old", "18-year-old", "40-year-old", "70-year-old"]
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

SEX_YOUNG = {"man": "boy", "woman": "girl"}

prompt_enhancements = {
    "cinematic still": "(cinematic still) of {prompt} . high quality, detailed photograph, (visible face:0.8)",
}


def construct_mugshot_prompt(sex, age, ethnicity):
    if age == "5-year-old":
        sex = SEX_YOUNG[sex]
    return f"mugshot of a {age} {ethnicity} {sex}, detailed, high quality, full face photo, realistic, front facing, neutral background"


def get_id_code(sex, age, ethnicity):
    age_code = age.split("-")[0]
    sex_code = sex[:3]
    ethnicity_code = ethnicity[:3]
    return f"{age_code}-{ethnicity_code}-{sex_code}"


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


def process_file(file_path):
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
