import os

script_dir = os.path.dirname(__file__)

ROOT_DIR = os.path.abspath(os.path.join(script_dir, ".."))

DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
