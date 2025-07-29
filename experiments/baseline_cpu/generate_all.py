import jinja2

from pyrlmala.utils.posteriordb import PosteriorDBToolbox

posteriordb_path = "../posteriordb/posterior_database"
pdb_toolbox = PosteriorDBToolbox(posteriordb_path)


pdb_model_name_list = [
    i for i in pdb_toolbox.get_model_name_with_gold_standard() if "test" not in i
]

for model_name in pdb_model_name_list:
    with open("templete/baseline_run.py", "r") as f_py:
        py_template = jinja2.Template(f_py.read())

    with open("templete/run-baseline.sh", "r") as f_bash:
        bash_template = jinja2.Template(f_bash.read())

    py_context = {"model_name": model_name, "posteriordb_path": posteriordb_path}
    with open(f"baseline_run_{model_name}.py", "w") as f_py:
        f_py.write(py_template.render(py_context))

    bash_context = {"model_name": model_name}
    with open(f"run-baseline_{model_name}.sh", "w") as f_bash:
        f_bash.write(bash_template.render(bash_context))
