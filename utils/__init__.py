import io
import os
import traceback

from .sequential_timer import SequentialTimer


def get_exception_traceback_str(exc: Exception) -> str:
    file = io.StringIO()
    traceback.print_exception(exc, file=file)
    return file.getvalue().rstrip()


def list_directories_in_directory(directory_path):
    try:
        # List all entries in the given directory
        entries = os.listdir(directory_path)

        # Filter out entries that are directories
        directories = [
            entry
            for entry in entries
            if os.path.isdir(os.path.join(directory_path, entry))
        ]

        return directories
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
