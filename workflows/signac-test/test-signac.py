import signac
from funnel_cake import project_path

project = signac.init_project(name="TestSignacProject", root=project_path("signac-test"))

def V_idg(N, kT, p):
    return N * kT / p

for p in 0.1, 1.0, 10.0:
    sp = {"p": p, "kT": 1.0, "N": 1000} # state point
    job = project.open_job(sp)
    job.document["V"] = V_idg(**sp) # unwrap statepoint
    V = V_idg(**job.statepoint())
    with job: # cd into job workspace
        with open("V.txt", "w") as file:
            file.write(str(V) + "\n")

print(project.workspace())

# job = project.open_job({"p": 1.0, "kT": 1.0, "N": 1000})

# print(job.statepoint())
# print(job.workspace())