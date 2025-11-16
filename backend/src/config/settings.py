import yaml
from pathlib import Path

def get_settings():
    """Load the settings from the YAML file."""
    config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
    try:
        with open(config_path, 'r') as file:
            settings = yaml.safe_load(file)
        return settings
    except Exception as e:
        # Handle exceptions (e.g., file not found, YAML parsing error)
        print(f"Error loading settings: {e}")
        return {}
