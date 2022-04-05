import os
import pathlib

CAKE_DATA_DIR = pathlib.Path(os.environ["CAKE_DATA_DIR"]) / "funnel-cake"

def project_path(dir: str):
    assert dir != ""
    return CAKE_DATA_DIR / dir