import signac
from time import sleep
from funnel_cake import project_path
import numpy as np

project = signac.get_project(project_path("signac-test"))

def V_vdW(p, kT, N, a=0, b=0):
    """Solve the van der Waals equation for V."""
    coeffs = [p, -(kT * N + p * N * b), a * N ** 2, -a * N ** 3 * b]
    V = sorted(np.roots(coeffs))
    return np.real(V).tolist()

# for job in project:
#     if "a" not in job.sp:
#         job.sp.a = 0
#     if "b" not in job.sp:
#         job.sp.b = 0

for job in project:
    if "V" in job.document:
        job.document["V_liq"] = 0
        job.document["V_gas"] = job.document.pop("V")
        with open(job.fn("V.txt"), "w") as file:
            file.write("{},{}\n".format(0, job.document["V_gas"]))

for job in project:
    print(job.statepoint(), job.document)

vdW = {
    # Source: https://en.wikipedia.org/wiki/Van_der_Waals_constants_(data_page)
    "ideal gas": {"a": 0, "b": 0},
    "argon": {"a": 1.355, "b": 0.03201},
    "water": {"a": 5.536, "b": 0.03049},
}

def calc_volume(job):
    V = V_vdW(**job.statepoint())
    job.document["V_liq"] = min(V)
    job.document["V_gas"] = max(V)
    with open(job.fn("V.txt"), "w") as file:
        file.write(f"{min(V)},{max(V)}\n")

for fluid in vdW:
    for p in np.linspace(0.1, 10.0, 10):
        sp = {"N": 1000, "p": float(p), "kT": 1.0}
        sp.update(vdW[fluid])
        job = project.open_job(sp)
        job.document["fluid"] = fluid
        calc_volume(job)

ps = {job.statepoint()["p"] for job in project}
for fluid in sorted(vdW):
    print(fluid)
    for p in sorted(ps):
        jobs = project.find_jobs({"p": p}, doc_filter={"fluid": fluid})
        for job in jobs:
            print(
                round(p, 2),
                round(job.document["V_liq"], 4),
                round(job.document["V_gas"], 2),
            )
    print()

