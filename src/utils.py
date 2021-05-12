import tempfile
from typing import Dict, List


def split_to_temp(split: List) -> str:
    f = tempfile.NamedTemporaryFile(mode="w+", newline="\n")
    for line in split:
        f.write(line)
    f.close()
    return f.name


def dataset_to_temp(dataset: Dict) -> (str, str, str):
    return (
        split_to_temp(dataset["train"]),
        split_to_temp(dataset["validation"]),
        split_to_temp(dataset["test"]),
    )
