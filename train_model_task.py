from datasets import load_dataset
from src.train_model import train_model
from clearml import Task, Dataset

Task.add_requirements("datasets")
Task.add_requirements("transformers[torch]")
Task.add_requirements("tqdm")
task = Task.init(project_name="password-model", task_name="train_model")

args = {
    "dataset_task_id": ""
}

task.connect(args)

task.execute_remotely()

# get dataset from task's artifact
if args['dataset_task_id']:
    dataset_task = Task.get_task(task_id=args['dataset_task_id'])
    train_local_path = dataset_task.artifacts["train"].get_local_copy()
    validation_local_path = dataset_task.artifacts["validation"].get_local_copy()
    test_local_path = dataset_task.artifacts["test"].get_local_copy()
else:
    raise ValueError("Missing dataset link")

dataset = load_dataset(
    "text",
    data_files={
        "train": train_local_path,
        "test": validation_local_path,
        "validation": test_local_path,
    }
)

model = train_model(dataset)

task.upload_artifact("model", artifact_object=model)
