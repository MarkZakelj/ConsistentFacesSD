import argparse
import json

from workflow_builder.workflow_utils import replace_key


def main():
    parser = argparse.ArgumentParser(
        description="Compile default comfy workflow to Katalist comfy workflow"
    )
    parser.add_argument("-f", "--file", help="Target file", required=True)
    parser.add_argument("-o", "--output", help="Output file", required=True)
    parser.add_argument(
        "-n",
        "--numbered",
        help="ID are numbers instead of string names",
        action="store_true",
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--load-image64",
        help="Transform LoadImage nodes to LoadImageBase64 nodes",
        action="store_true",
    )
    # group.add_argument(
    #     "--load-image", help="Transform LoadImage/LoadMask Base64 nodes to LoadImage nodes", action="store_true"
    # )
    args = parser.parse_args()
    if args.output == args.file:
        raise ValueError("Output file should be different from the input file")
    if not args.output.endswith(".json"):
        raise ValueError("Output file should be a json file")
    target = json.load(open(args.file))
    # turn all keys into numbers to avoid conflicts
    for i, key in enumerate(list(target.keys())):
        target = replace_key(key, str(i), target)

    # construct names for nodes from titles - custom titles set in comfy will be reflected here
    names = {key: val["_meta"]["title"].replace(" ", "") for key, val in target.items()}
    numerated_names = set()
    # enumerate keys and rename workflow keys - avoid same key-names in the workflow
    for key, val in list(target.items()):
        raw_name = names[key]
        if raw_name == "PreviewImage":
            target.pop(key)
            continue
        num = 0
        while raw_name + str(num) in numerated_names:
            num += 1
        enumerated_name = raw_name + str(num)
        numerated_names.add(enumerated_name)
        names[key] = enumerated_name

    for key, val in list(target.items()):
        target[names[key]] = target[key]
        for input_key, input_val in val["inputs"].items():
            if (
                isinstance(input_val, list)
                and len(input_val) == 2
                and isinstance(input_val[0], str)
            ):
                if input_val[0] in names:
                    input_val[0] = names[input_val[0]]

    # while dq:
    #     key = dq.popleft()
    #     target[names[key]] = target[key]
    #     for input_key, val in target[key]["inputs"].items():
    #         if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str) and val[0] in names:
    #             dq.append(val[0])
    #             val[0] = names[val[0]]
    for key in names:
        if key in target and key != names[key]:
            del target[key]

    # replace LoadImage nodes with LoadImage(Base64) or LoadMask(Base64) nodes
    if args.load_image64:
        i = 0
        while f"LoadImage{i}" in target:
            target[f"LoadImage(Base64){i}"] = {
                "inputs": {"image": ""},
                "class_type": "ETN_LoadImageBase64",
                "_meta": {"title": "Load Image (Base64)"},
            }
            target[f"LoadMask(Base64){i}"] = {
                "inputs": {"mask": ""},
                "class_type": "ETN_LoadMaskBase64",
                "_meta": {"title": "Load Mask (Base64)"},
            }
            has_image = False
            has_mask = False

            # iterate over all keys in the workflow
            for key, val in target.items():
                # if the value is a dictionary and has "inputs" key
                if isinstance(val, dict) and "inputs" in val:
                    # iterate over all keys in the inputs dictionary - where the inputs to the node are defined
                    for key2, val2 in val["inputs"].items():
                        # if the value is a list and has length of 2 (the input is another node)
                        # and the first element is the LoadImage node
                        if (
                            isinstance(val2, list)
                            and len(val2) == 2
                            and val2[0] == f"LoadImage{i}"
                        ):
                            # change the input to be LoadImage(Base64) node
                            print(val["inputs"][key2])
                            if val["inputs"][key2][1] == 0:
                                val["inputs"][key2] = [f"LoadImage(Base64){i}", 0]
                                has_image = True
                            elif val["inputs"][key2][1] == 1:
                                val["inputs"][key2] = [f"LoadMask(Base64){i}", 0]
                                has_mask = True
                            else:
                                raise ValueError("Invalid input index")
            if not has_image:
                del target[f"LoadImage(Base64){i}"]
            if not has_mask:
                del target[f"LoadMask(Base64){i}"]
            del target[f"LoadImage{i}"]
            i += 1
    # Custom constant rename of node keys
    # for old, new in RENAME_MAP.items():
    #     for key in list(target.keys()):
    #         if key.startswith(old):
    #             num = key.lstrip(old)
    #             target = replace_key(key, new + num, target)
    # for key, val in list(target.items()):
    #     if val["_meta"]["title"].endswith("---"):
    #         target = replace_key(key, val["_meta"]["title"].rstrip("---"), target)
    # renumber_workflow(target, True)
    json.dump(target, open(args.output, "w"), indent=4)


if __name__ == "__main__":
    main()
