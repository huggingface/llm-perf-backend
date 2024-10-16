from dotenv import load_dotenv
import os

load_dotenv()


def is_debug_mode():
    return os.environ.get("DEBUG_MODE", "0") == "1"
