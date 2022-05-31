import signac
from time import sleep
from funnel_cake import project_path
import numpy as np

project = signac.get_project(project_path("signac-test"))

def V_idg(N, kT, p, cost=0):
    sleep(cost)
    return N * kT / p

def compute_volume(job: signac.Project.Job):
    print("Computing volume of", job)
    V = V_idg(cost=1, **job.statepoint())
    job.document["V"] = V
    with open(job.fn("V.txt"), "w") as file:
        file.write(str(V) + "\n")


# for job in project:
#     compute_volume(job)

def init_statepoints(n):
    for p in np.linspace(0.1, 10.0, n):
        sp = {"p": float(p), "kT": 1.0, "N": 1000}
        job = project.open_job(sp)
        job.init()
        print("initialize", job)


# init_statepoints(10)

def classify(job: signac.Project.Job):
    yield "init"
    if "V" in job.document and job.isfile("V.txt"):
        yield "volume-computed"

print(f"Status: {project}")
for job in project:
    labels = ", ".join(classify(job))
    p = round(job.sp.p, 1)
    print(job, p, labels)

for job in project:
    labels = classify(job)
    if "volume-computed" not in labels:
        compute_volume(job)