from clearml import Task


# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name="password-model", task_name="Test task")

task.execute_remotely()

print("Doing something")
print("Done")