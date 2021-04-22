import os
from datasets import load_dataset
from src.train_model import train_model
from clearml import Task, Dataset

Task.add_requirements("datasets")
Task.add_requirements("transformers[torch]")
Task.add_requirements("tqdm")
task = Task.init(project_name = "password-model", task_name = "train_model")

task.execute_remotely()

prepared_dataset = Dataset.get(dataset_project = "password-model", dataset_name = "rockyou_prepared")
data_dir = prepared_dataset.get_local_copy("data/rockyou/")
print(data_dir)

dataset = load_dataset(
    "text", 
    data_files = {
        "train": os.path.join(data_dir, "train.txt"),
        "test": os.path.join(data_dir, "test.txt"),
        "validation": os.path.join(data_dir, "validation.txt"),
    }
)

model = train_model(dataset)



