from clearml import Task
from clearml.automation.controller import PipelineController

# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name='password-model', task_name='base-pipeline',
                 task_type=Task.TaskTypes.controller, reuse_last_task_id=False)

pipe = PipelineController(default_execution_queue='default', add_pipeline_tags=False)
pipe.add_step(name="stage_prepare_dataset", base_task_project="password-model", base_task_name="prepare_dataset")
pipe.add_step(name="stage_train_model", base_task_project="password-model", base_task_name="train_model",
              parents=["stage_prepare_dataset"],
              parameter_override={"General/dataset_task_id": "${stage_prepare_dataset.id}"})

pipe.start()
pipe.wait()
pipe.stop()
