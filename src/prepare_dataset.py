import os
import json
from sklearn.model_selection import train_test_split
from typing import Optional

def train_val_test(dataX, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2):
    x_train, x_test = train_test_split(dataX, test_size=1 - train_ratio)
    x_val, x_test = train_test_split(x_test, test_size=test_ratio/(test_ratio + validation_ratio)) 
    return {
        "train": x_train,
        "validation": x_val,
        "test": x_test
    }

def prepare_dataset(dataset_filename: str, n_limit: Optional[int] = None, train_ratio: float = 0.6, validation_ratio: float = 0.2, test_ratio: float = 0.2, output_dir: str = "data/dataset"):
    with open(dataset_filename, "r", encoding="utf8", errors="ignore") as f:
        if n_limit is not None and n_limit != -1:
            passwords = list(f)[:n_limit]
        else:
            passwords = list(f)

    dataset = train_val_test(passwords, train_ratio, validation_ratio, test_ratio)

    print(f"Full dataset size: {len(passwords)}")
    print(f"Train size: {len(dataset['train'])}")
    print(f"Validation size: {len(dataset['validation'])}")
    print(f"Test size: {len(dataset['test'])}")

    os.makedirs(output_dir, exist_ok=True)
    for k in dataset:
        with open(f"{output_dir}/{k}.txt", "w") as f:
            for row in dataset[k]:
                f.write(row)