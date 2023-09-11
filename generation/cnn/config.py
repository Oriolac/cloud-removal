import yaml
from typing import Any, Dict


class Config:
    """Base class for configuration."""

    def __init__(self, **entries: Dict[str, Any]):
        self.__dict__.update(entries)


def generate_class_from_dict(d: Dict[str, Any], class_name: str) -> Config:
    """Generate a class dynamically from a dictionary."""
    new_class = Config(**d)
    for k, v in d.items():
        if isinstance(v, dict) and k != "kwargs":
            setattr(new_class, k, generate_class_from_dict(v, k.capitalize()))
    return new_class

if __name__ == '__main__':
    # Read the YAML and generate classes
    with open('configs/config_cnn.yml', 'r') as file:
        yaml_content = yaml.safe_load(file)

    config = generate_class_from_dict(yaml_content, "Config")

    # Function to pretty print the class attributes
    def pretty_print(obj: Config, indent: int = 0):
        print(obj)
        for attr, value in obj.__dict__.items():
            if isinstance(value, Config):
                print("  " * indent + f"{attr}:")
                pretty_print(value, indent + 1)
            else:
                print("  " * indent + f"{attr}: {value}")


    # Pretty print the generated classes
    pretty_print(config)
