import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "../datasets/")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs")