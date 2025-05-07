import os
import yaml
from pathlib import Path

# Define base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
CONFIG_DIR = os.path.join(BASE_DIR, "config")

# Ensure directories exist
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

def load_config(config_file: str) -> dict:
    """Load configuration from a YAML file."""
    config_path = os.path.join(CONFIG_DIR, config_file)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Virhe konfiguraation lataamisessa ({config_file}): {str(e)}")
        return None