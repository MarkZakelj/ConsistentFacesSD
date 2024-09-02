import json
import os

from utils.paths import DATA_DIR


def main():
    for n in [1, 2, 3]:
        filepath = os.path.join(DATA_DIR, f"prompt_seed_pairs_{n}.json")
        ps_pairs = json.load(open(filepath, "r"))
        for i, pair in enumerate(ps_pairs):
            del pair["id"]
            pair["img_code"] = i
        json.dump(ps_pairs, open(filepath, "w"), indent=2)


if __name__ == "__main__":
    main()
