import os
import random

random.seed(42)

ages = ["5 year old", "18 year old", "40 year old", "70 year old"]
ethnicities = ["caucasian", "black", "asian", "hispanic", "arab", "indian", "native american"]
sexes = ["man", "woman"]

sex_young = {
    'man': 'boy',
    'woman': 'girl'
}


def replace_with_random_person(line):
    sex = random.choice(sexes)
    age = random.choice(ages)
    ethnicity = random.choice(ethnicities)
    if age == '5 year old':
        sex = sex_young[sex]
    line = line.replace("person", f"{age} {ethnicity} {sex}")
    return line


def process_file(file_path):
    processed_lines = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            assert line.startswith("person"), "First word of each line must be 'person'"
            processed_line = "cinematic still of a " + line.strip() + \
                             ", high quality, detailed photograph"
            # line_with_person_replaced = replace_person(processed_line)
            processed_lines.append(processed_line)
    return processed_lines


def main():
    file_path = os.path.join("data", "raw_prompts.txt")
    processed_lines = process_file(file_path)
    for line in processed_lines:
        print(line)


if __name__ == '__main__':
    main()
