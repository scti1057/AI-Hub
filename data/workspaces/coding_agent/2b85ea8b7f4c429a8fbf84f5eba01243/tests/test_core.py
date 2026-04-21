import os
import json
from src.task_manager.storage import load_tasks, save_tasks, get_next_id, STORAGE_FILE


def test_storage_lifecycle():
    """Tests the basic storage lifecycle: save, load, and ID generation."""
    # Clean up any existing file
    if os.path.exists(STORAGE_FILE):
        os.remove(STORAGE_FILE)
    
    # Test ID generation
    assert get_next_id([]) == 1
    assert get_next_id([{"id": 1}]) == 2
    assert get_next_id([{"id": 1}, {"id": 3}]) == 4
    
    # Test saving tasks
    tasks = [{"id": 1, "title": "Task 1"}]
    save_tasks(tasks)
    
    # Test loading tasks
    loaded_tasks = load_tasks()
    assert len(loaded_tasks) == 1
    assert loaded_tasks[0]["id"] == 1
    assert loaded_tasks[0]["title"] == "Task 1"
    
    # Test saving more tasks with auto IDs
    new_task = {"title": "Task 2"}
    tasks.append(new_task)
    save_tasks(tasks)
    
    loaded_tasks = load_tasks()
    assert len(loaded_tasks) == 2
    assert loaded_tasks[1]["id"] == 2
    assert loaded_tasks[1]["title"] == "Task 2"
    
    # Clean up
    os.remove(STORAGE_FILE)
