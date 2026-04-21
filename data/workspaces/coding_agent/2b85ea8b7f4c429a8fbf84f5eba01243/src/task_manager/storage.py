import json
import os

STORAGE_FILE = "tasks.json"


def get_next_id(tasks):
    """Returns the next available task ID, starting from 1."""
    if not tasks:
        return 1
    return max(task["id"] for task in tasks) + 1


def load_tasks():
    """Loads tasks from the storage file."""
    if not os.path.exists(STORAGE_FILE):
        return []
    with open(STORAGE_FILE, "r") as f:
        return json.load(f)


def save_tasks(tasks):
    """Saves tasks to the storage file."""
    with open(STORAGE_FILE, "w") as f:
        json.dump(tasks, f, indent=2)
