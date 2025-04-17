import os

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_ID2PATH = {
    "qwen2.5-3b": "/opt/tiger/models/Qwen2.5-3B-Instruct",
    "qwen2.5-7b": "/opt/tiger/models/Qwen2.5-7B-Instruct",
}

MODEL_ID2GATE = {
    "qwen2.5-3b": "/opt/tiger/seer/seer_attn_qwen2.5-3B",
    "qwen2.5-7b": "/opt/tiger/seer/seer_attn_qwen2.5-7B",
}

DATA_ID2PATH = {
    "wiki": f"/opt/tiger/datasets/wikitext",
    "pg19": f"/opt/tiger/datasets/emozilla--pg19-test",
    "redpajama": f"/opt/tiger/datasets/togethercomputer--RedPajama-Data-1T-Sample",
}

DATA_DIR = f"{PROJ_DIR}/datasets"

