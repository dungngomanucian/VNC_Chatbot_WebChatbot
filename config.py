import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME_HF = "meta-llama/Llama-3.2-1B-Instruct"

MODEL_PATH_LOCAL = None
USE_LLAMA_CPP = False
MODEL_PATH_CPP = None

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

FORCE_CPU = False

GRADIO_SERVER_NAME = "localhost"
GRADIO_SERVER_PORT = 7860
GRADIO_SHARE = False

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
