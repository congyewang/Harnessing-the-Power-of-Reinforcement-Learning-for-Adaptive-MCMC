import jinja2
from pyrlmala.utils import Toolbox


# dir_name = [f"laplace_{i}" for i in [1, 2, 4]]
# model_name_list = [f"test-laplace_{i}-test-laplace_{i}" for i in [1, 2, 4]]
dir_name = ["banana", "neals_funnel"]
model_name_list = [
    "test-banana-test-banana",
    "test-neals_funnel-test-neals_funnel",
]
step_size_list = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]


py_template_path = "./template/template.const_run.py"
bash_template_path = "./template/template.run-const.sh"

with open(py_template_path, "r") as file:
    py_template = jinja2.Template(file.read())

with open(bash_template_path, "r") as file:
    bash_template = jinja2.Template(file.read())

for idx, model_name in enumerate(model_name_list):
    for step_size in step_size_list:
        context = {
            "model_name": model_name,
            "step_size": step_size,
        }
        config_content = py_template.render(context)

        py_output_path = f"./{dir_name[idx]}/const/const_run_{step_size}.py"
        Toolbox.create_folder(py_output_path)

        with open(py_output_path, "w") as file:
            file.write(config_content)

        bash_output_path = f"./{dir_name[idx]}/const/run-const_{step_size}.sh"
        with open(bash_output_path, "w") as file:
            file.write(bash_template.render(context))
