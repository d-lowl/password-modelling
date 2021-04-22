from clearml import Task, Dataset
import os
from src.prepare_dataset import prepare_dataset

task = Task.init(project_name = "password-model", task_name = "prepare_password_dataset", task_type = Task.TaskTypes.data_processing)
Task.add_requirements("sklearn")

task.execute_remotely()

raw_dataset = Dataset.get(dataset_project = "password-model", dataset_name = "rockyou")

dataset_dir = raw_dataset.get_mutable_local_copy("clearml_data/", overwrite = True)

prepare_dataset(
    dataset_filename = os.path.join(dataset_dir, "rockyou.txt"),
    output_dir = os.path.join(dataset_dir, "dataset/"),
    n_limit = 10000
)

dataset = Dataset.create(dataset_project = "password-model", dataset_name = "rockyou_prepared", parent_datasets = [raw_dataset], use_current_task = True)

dataset.add_files(os.path.join(dataset_dir, "dataset/"), verbose=True)
dataset.upload()
dataset.finalize()