import os
import random

random.seed(42)

ages = ["5-year-old", "18-year-old", "40-year-old", "70-year-old"]
ethnicities = [
    "black",
    "asian",
    "hispanic",
    "arab",
    "indian",
    "native american",
    "caucasian",
]
sexes = ["man", "woman"]

sex_young = {"man": "boy", "woman": "girl"}

prompt_enhancements = {
    "cinematic still": "(cinematic still) of {prompt} . high quality, detailed photograph, (visible face:0.8)",
}


def replace_with_random_person(line, person_code):
    sex = random.choice(sexes)
    age = random.choice(ages)
    ethnicity = random.choice(ethnicities)
    if age == "5-year-old":
        sex = sex_young[sex]
    if person_code not in line:
        raise ValueError(f"Person code {person_code} not found in line")
    line = line.replace(person_code, f"{age} {ethnicity} {sex}")
    return line


def enchance_prompt(prompt: str, style_name: str):
    pass


def process_file(file_path):
    processed_lines = []
    with open(file_path, "r") as file:
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
