from pathlib import Path
import pickle as p


def get_project_root() -> Path:
    return Path(__file__).parent


def pickle_save(obj, filename):
    with open(filename, "wb") as f:
        p.dump(obj, f)


def pickle_load(filename):
    with open(filename, "rb") as f:
        return p.load(f)
