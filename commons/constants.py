# data directory paths
MAIR_LAB_DATA_DIR = "MAIR_LAB_DATA_DIR/coco/input_dataset"
SYNTH_DIFFUSE_DATA_DIR = "SYNTH_DIFFUSE_DATA_DIR/generated_question_for_verification"
VIM_DATA_DIR = "VIM_DATA_DIR/generated_output_image"
COCO_DATA_DIR = "COCO_DATA_DIR/coco/input_dataset"

TOTAL_NUM_COCO_CHUNKS = 15

# api keys - loaded from environment variables
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_key")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_gemini_api_key")

# checkpoints
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints")
LAMA_CKPT = os.getenv("LAMA_CKPT", "models/big-lama")
LAMA_CONFIG = os.getenv("LAMA_CONFIG", "lama/configs/prediction/default.yaml")

LANGUAGE_MODEL_NAMES = [
    "google/gemma-7b",
    "google/gemma-7b-it",
    "lmsys/vicuna-13b-v1.3",
    "lmsys/vicuna-13b-v1.5",
    "lmsys/vicuna-33b-v1.3",
    "Open-Orca/OpenOrca-Platypus2-13B",
    "Open-Orca/Mistral-7B-OpenOrca",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "distilroberta-base",
    "TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ",
    "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
    "teknium/OpenHermes-2.5-Mistral-7B",
    "gemini-2.5-flash-lite",
]
MISTRALAI_LANGUAGE_MODEL_NAMES = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ",
    "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
]

# VALID_SPATIAL_DIRECTIONS = ["left", "right", "top", "bottom", "below", "above", "under"]
VALID_SPATIAL_DIRECTIONS = ["above", "below", "under", "bottom"]
# VALID_COUNTS = ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
VALID_COUNTS = ["five", "six", "seven", "eight", "nine", "ten"]
SYNTH_ONLY_CATEGORIES = ["counting", "relation"]
VALID_CATEGORY_NAMES = ["object", "attribute", "relation", "counting"]
COCO_ONLY_CATEGORIES = ["object", "attribute"]