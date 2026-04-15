from ai_hub.memory.sqlite_db import init_db
from ai_hub.config import DB_PATH


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at: {DB_PATH}")