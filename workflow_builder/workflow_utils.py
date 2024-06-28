import json
import os
import re
from collections import deque
from copy import deepcopy

from .const import IMG_EXTENSIONS
from .errors import NodeNotFoundError

END_KEY = "SendImage(WebSocket)"
END_KEY_NUMBERED = "SendImage(WebSocket)0"

# get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# get the workflow directory
workflow_dir = os.path.join(current_dir, "workflows")
compiled_workflow_dir = os.path.join(current_dir, "compiled_workflows")


def is_image_filename(string: str):
    """determine whether string is image filename"""
    for ext in IMG_EXTENSIONS:
        if string.endswith(ext):
            return True
    return False


def remove_number_from_end(s):
    """
    removes number from the end of string - used for workflow keys
    Args:
        s: string representing full workflow key with number at the end

    Returns:

    """
    return re.sub(r"\d+$", "", s)


def replace_key(old_key: str, new_key: str, workflow: dict, inline=False) -> dict:
    """
    Replace key in the workflow with a new key
    """
    if not inline:
        workflow = deepcopy(workflow)
    if new_key in workflow:
        raise ValueError(f"Key {new_key} already exists in workflow")
    for key, val in list(workflow.items()):
        if key == old_key:
            workflow[new_key] = val
            workflow.pop(old_key)
        else:
            for key2, val2 in val["inputs"].items():
                if isinstance(val2, list) and len(val2) == 2 and val2[0] == old_key:
                    val["inputs"][key2][0] = new_key
            workflow[key] = val
    return workflow


def link_after(
    first_node: str, second_node: str, links: list[tuple[int, str]], workflow: dict
) -> None:
    """
    Link the after_node after the before_node node, INLINE
    Args:
        first_node: name of the first node
        second_node: name of the second node
        links: list of tuples [(first_node_output_id: int, second_node_input_name: str), ...]
        workflow: workflow in dictionary form
    Returns: new workflow with the new node linked after the target node
    """
    if first_node not in workflow:
        raise NodeNotFoundError(first_node)
    if second_node not in workflow:
        raise NodeNotFoundError(second_node)
    for first_out, second_in in links:
        workflow[second_node]["inputs"][second_in] = [first_node, first_out]


def link_after_and_before(
    before_node: str,
    middle_node: str,
    links: list[tuple[int, str, int]],
    workflow: dict,
):
    """
    Link the middle node after the before node, and link middle_node before the nodes, the target node is connected to
    (replace connections
    WORKS INLINE
    Args:
        before_node: name of the target node
        middle_node: name of the middle node
        links: list of tuples [(before_node_out: int, middle_node_input: str, middle_node_out: int), ...] -
            outputs are listed as orders (numbers), but inputs are named
        workflow: workflow in dictionary form
    Returns: new workflow with the new node linked in front of the target node - hanging node linked properly
    """
    digraph = build_digraph(workflow)
    if before_node not in digraph:
        raise ValueError(f"Node {before_node} not found in the workflow")
    if middle_node not in digraph:
        raise ValueError(
            f"Node {middle_node} not found in the workflow - maybe merge the partial workflow first"
        )
    previous_links = digraph[before_node]["outputs"]
    for out_num, inp, out_num_middle in links:
        workflow[middle_node]["inputs"][inp] = [before_node, out_num]
        for node_name, (out_target, next_inp_name) in previous_links.items():
            if out_num == out_target and node_name != middle_node:
                workflow[node_name]["inputs"][next_inp_name] = [
                    middle_node,
                    out_num_middle,
                ]


def remove_inputs(node_name: str, inputs: list[str], workflow: dict):
    """
    Remove the inputs from the node. Used for dynamically changing Nodes like "Make Image List",
    by setting them to None
    Args:
        node_name: name of the node to remove the inputs from
        inputs: list of inputs to remove
        workflow: workflow in dictionary form
    """
    if node_name not in workflow:
        raise NodeNotFoundError(node_name)
    for inp in inputs:
        workflow[node_name]["inputs"][inp] = None


def skip_node(node_name: str, links: list[tuple[str, int]], workflow: dict):
    """
    Skip the node from the workflow and link the nodes connected to the dropped node, works INLINE
    For the outputs not specified in the links, the connected output nodes will delete the input key on the connection
    to this node
    Args:
        node_name: name of the node to be dropped
        links: list of tuples [(nodes_input: str, nodes_output: int), ...] - connected input and outputs of the node
            used to connect left and right nodes of the dropped node,
            for example LoraLoaderStack has "model" on the input,
            and "model" on the output on the 0th index, so the link would be [("model", 0)]
        workflow: workflow in dictionary form
    """
    # TODO - remove inputs of the after nodes if they are not connected to the node_name node
    digraph = build_digraph(workflow)
    if node_name not in digraph:
        raise ValueError(f"Node {node_name} not found in the workflow")
    # unlinked = dict(digraph[node_name]["outputs"].items())
    for inp, out in links:
        after_nodes = []
        for node_name2, vals in digraph[node_name]["outputs"].items():
            for val in vals:
                if val[0] == out:
                    after_nodes.append((node_name2, val[1]))
                    # unlinked.pop(node_name2, None)
        if len(after_nodes) == 0:
            raise ValueError(
                f"Node {node_name} not connected to the node with output {out}"
            )
        for after_node, after_node_input in after_nodes:
            workflow[after_node]["inputs"][after_node_input] = workflow[node_name][
                "inputs"
            ][inp]


def build_digraph(workflow):
    """
    Build a directed graph of the workflow - directed left and right
    nodes in Comfy workflows only have defined inputs,
    this also adds the outputs of the nodes
    Args:
        workflow: workflow dictionary
    Returns: workflow graph in the form of adjacency list
    """
    graph = {key: {"inputs": {}, "outputs": {}} for key in workflow.keys()}
    for key, val in workflow.items():
        for link, link_val in val["inputs"].items():
            if isinstance(link_val, list) and len(link_val) == 2:
                graph[key]["inputs"][link] = link_val
                if link_val[0] not in graph:
                    graph[link_val[0]] = {"inputs": {}, "outputs": {}}
                if key not in graph[link_val[0]]["outputs"]:
                    graph[link_val[0]]["outputs"][key] = []
                graph[link_val[0]]["outputs"][key].append([link_val[1], link])
    return graph


def load_workflow(
    workflow_name: str, compiled: bool = False, abspath: bool = False
) -> dict:
    """
    Load workflow from json file
    Args:
        workflow_name: name (with .json extension) of the workflow to be loaded
        compiled: should you search the compiled_workflows folder or not
        abspath: should you search in the current working directory or not
    Returns: workflow dictionary
    """
    target_dir = workflow_dir
    match (compiled, abspath):
        case (_, True):
            target_dir = "."
        case (True, _):
            target_dir = compiled_workflow_dir
        case (_, _):
            target_dir = workflow_dir
    return json.load(open(os.path.join(target_dir, workflow_name), "r"))


def get_multiples(workflow: dict) -> dict[str, list[str]]:
    vals = {}
    for key, val in workflow.items():
        # remove number from the end of string
        raw_name = remove_number_from_end(key)
        if not raw_name:
            raise ValueError(f"workflow not compiled properly: {key}")
        if raw_name not in vals:
            vals[raw_name] = []
        vals[raw_name].append(key)
    for key, val in list(vals.items()):
        if len(val) == 1:
            del vals[key]
    return vals


def get_same_class(workflow: dict, node_class) -> dict[str, dict]:
    result = {}
    for key, val in workflow.items():
        if key.startswith(node_class):
            raw_name = remove_number_from_end(key)
            if raw_name not in result:
                result[raw_name] = {}
            result[raw_name][key] = deepcopy(val)
    return result


def merge_partial_workflow(
    workflow: dict, part_workflow_name: str, overwrite: list = None
) -> dict:
    """
    Merge partial workflow into the main workflow
    Partial workflows are workflows that are not complete and are meant to be merged into the main workflow, e.g
    face.json, lora.json, ...
    basic.json is a standalone workflow and can be used by itself
    """
    partial_workflow = load_workflow(part_workflow_name)
    # validate merge - no duplicate keys
    for key in partial_workflow:
        if key in workflow and key not in overwrite:
            raise ValueError(f"Duplicate key found in partial workflow: {key}")
        workflow[key] = partial_workflow[key]
    return workflow


def trim_workflow(workflow: dict, inline=False) -> dict:
    """
    Trim the workflow to remove unnecessary keys, not connected (directly or indirectly) to the end node
    Args:
        workflow: workflow to be trimmed
        inline: create a new workflow or modify the existing one
    Returns: trimmed workflow without hanging nodes
    hangin nodes are nodes not connected to the end node in a directed graph - leaves
    Examples:
    =====================================
    [Start] -> [Node1] -> [Node2] -> [End]
    all nodes are connected to the end node
    =====================================
    [Start] -> [Node1] -> [Node2] -> [End]
                       |-> [Node3]
    Node3 is a hanging node and will be removed by trimming
    """
    # use BFS to find all nodes connected to the end node
    dq = deque([END_KEY_NUMBERED])
    new_workflow = {}
    while dq:
        key = dq.popleft()
        new_workflow[key] = workflow[key]
        for input_key, val in workflow[key]["inputs"].items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                dq.append(val[0])
    if inline:
        for key in list(workflow.keys()):
            if key not in new_workflow:
                del workflow[key]
    return new_workflow


def remove_base64_images(workflow: dict) -> dict:
    """
    Remove base64 images from the workflow by changing their value to empty string
    Do this when you want to look at the workflow, and you don't want to see the long base64 image strings
    """
    new_workflow = {}
    for key, val in workflow.items():
        if "base64" in key.lower():
            if "image" in val["inputs"]:
                val["inputs"]["image"] = ""
            if "mask" in val["inputs"]:
                val["inputs"]["mask"] = ""
        new_workflow[key] = val
    return new_workflow


def convert_loadb64_to_load(workflow: dict, target=None, inline=False) -> dict:
    """
    Convert LoadBase64 nodes to LoadImage nodes in the workflow
    useful for transferring the workflow to comfy and then testing
    If target is specified, only change the node with the target name
    """
    if not inline:
        workflow = deepcopy(workflow)
    for key, val in list(workflow.items()):
        if (target is None and "Base64" in key and "Load" in key) or (key == target):
            workflow[key] = {
                "inputs": {"image": "", "upload": "image"},
                "class_type": "LoadImage",
                "_meta": {"title": "Load Image"},
            }
    return workflow


def convert_load_to_loadb64(workflow: dict, target=None, inline=False) -> dict:
    """
    Convert LoadImage nodes to LoadImageBase64 nodes in the workflow
    useful for transferring the workflow to comfy and then testing
    If target is specified, only change the node with the target name
    """
    if not inline:
        workflow = deepcopy(workflow)
    for key, val in list(workflow.items()):
        if (target is None and "Base64" not in key and "Load" in key) or (
            key == target
        ):
            workflow[key] = {
                "inputs": {"image": ""},
                "class_type": "ETN_LoadImageBase64",
                "_meta": {"title": "Load Image (Base64)"},
            }
    return workflow
