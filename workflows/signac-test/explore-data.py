import signac
from funnel_cake import project_path

project = signac.get_project(project_path("signac-test"))

jobs_p_gt_0_1 = [job for job in project if job.sp.p > 0.1]
for job in jobs_p_gt_0_1:
    print(job.statepoint())

