from clearml import Task, Dataset
import os
from src.prepare_dataset import prepare_dataset

Task.add_requirements("sklearn")
task = Task.init(project_name="password-model", task_name="prepare_dataset",
                 task_type=Task.TaskTypes.data_processing)

task.execute_remotely()

raw_dataset = Dataset.get(dataset_project="password-model", dataset_name="rockyou")

dataset_dir = raw_dataset.get_mutable_local_copy("clearml_data/", overwrite=True)

train_f, validation_f, test_f = prepare_dataset(
    dataset_filename=os.path.join(dataset_dir, "rockyou.txt"),
    output_dir=os.path.join(dataset_dir, "dataset/"),
    n_limit=-1
)

task.upload_artifact("train", artifact_object=train_f)
task.upload_artifact("validation", artifact_object=validation_f)
task.upload_artifact("test", artifact_object=test_f)
