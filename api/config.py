import json
from typing import Any, Dict

def load_config(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as file:
        return json.load(file)

CONFIG = load_config("configs/config_rl.json")
