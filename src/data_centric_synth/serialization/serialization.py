import pickle as pkl
from pathlib import Path
from typing import Any


def save_to_pickle(obj: Any, path: Path):
    """Save the given object to a pickle file.
    Args:
        obj (Any): The object to save.
        path (Path): The path to save the object to.
    """
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open("wb") as f:
        pkl.dump(obj, f)


def load_from_pickle(path: Path) -> Any:
    """Load an object from a pickle file.
    Args:
        path (Path): The path to load the object from.
    Returns:
        Any: The loaded object.
    """
    with path.open("rb") as f:
        return pkl.load(f)
