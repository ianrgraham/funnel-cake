import signac
import flow
from funnel_cake import project_path
import numpy as np

project = signac.init_project(name="FlowTutorialProject", root=project_path("flow-test"))

def V_idg(N, p, kT):
    return N * kT / p

class MyProject(flow.FlowProject):
    pass

@MyProject.label
def estimated(job):
    return "V" in job.document

@MyProject.operation
@MyProject.post(estimated)
def compute_volume(job):
    job.document["V"] = V_idg(**job.statepoint())

for p in np.linspace(0.5, 5.0, 10):
    sp = dict(N=1728, kT=1.0, p=float(p))
    job = project.open_job(sp)
    print(job.id)
    job.init()

project = MyProject.get_project(root=project_path("flow-test"))

project.print_status(detailed=True)

project.run(num_passes=3)

project.print_status(detailed=True)

for job in project:
    print(job.statepoint()["p"], job.document.get("V"))

from matplotlib import pyplot as plt

V = {}

for job in project:
    V[job.statepoint()["p"]] = job.document["V"]

p = sorted(V.keys())
V = [V[p_] for p_ in p]
print(V)

plt.plot(p, V, label="idG")
plt.xlabel(r"pressure [$\epsilon / \sigma^3$]")
plt.ylabel(r"volume [$\sigma^3$]")
plt.legend()
plt.show()