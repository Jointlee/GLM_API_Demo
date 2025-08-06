import json
import os

from tqdm import tqdm
import yaml


def load_json_file(file_path: str) -> dict:
    """
    Reads a JSON file and returns the data as a dictionary.
    Args:
        file_path (str): The file path to the JSON file.
    Returns:
        dict: A dictionary containing the data from the JSON file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_json_file(data, path: str):
    """
    Saves the given data to a JSON file at the specified path.
    Args:
        data (dict or list): The data to be saved.
        path (str): The file path where the data should be saved.
    """
    if not path.lower().endswith(".json"):
        path = os.path.splitext(path)[0] + ".json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Successfully saved data to {path}")

def load_jsonl_file(file_path: str) -> list:
    """
    Reads a JSONL file and returns the data as a list of dictionaries, with a progress bar.
    Args:
        file_path (str): The file path to the JSONL file.
    Returns:
        list: A list of dictionaries containing the data from the JSONL file.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        total = len(lines)
        for line in tqdm(lines, total=total, desc="Loading JSONL"):
            data.append(json.loads(line.strip()))
    return data

def save_jsonl_file(data: list, path: str):
    """
    Saves the given data to a JSONL file at the specified path.
    Args:
        data (list): The data to be saved.
        path (str): The file path where the data should be saved.
    """
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Successfully saved data to {path}")

def save_tsv_file(data, path):
    """
    Saves the given data to a TSV file at the specified path.
    Args:
        data (list): The data to be saved. Each item should be a list or tuple representing a row.
        path (str): The file path where the data should be saved.
    """
    assert isinstance(data, list), "data must be a list"
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            assert isinstance(item, (list, tuple)), "Each item in data must be a list or tuple"
            f.write('\t'.join(str(x) for x in item) + "\n")
    print(f"Successfully saved data to {path}")

def load_tsv_file(path, skip_header=False):
    """
    Loads a TSV file and returns the data as a list of lists.
    Args:
        path (str): The file path to the TSV file.
        skip_header (bool): Whether to skip the first line (header) of the TSV file.
    Returns:
        list: A list of lists containing the data from the TSV file.
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        if skip_header:
            next(f)  # Skip the header line
        for line in f:
            data.append(line.strip().split('\t'))
    return data
