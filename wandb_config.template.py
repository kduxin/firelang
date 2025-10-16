from typing import Union
from pathlib import Path
import os
import getpass
import wandb

wandb_dir = f"/tmp/firelang_wandb_{getpass.getuser()}"
os.makedirs(wandb_dir, exist_ok=True)

settings = {
    "WANDB_ENTITY": "allen",  # replace this with your WANDB account name
    "WANDB_DIR": wandb_dir,
    # ----- cloud -----
    "WANDB_API_KEY": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  # replace this with your own WANDB API KEY
    "WANDB_PROJECT": "firelang",  # you can change this to the name you like
    "WANDB_BASE_URL": "https://api.wandb.ai/",
}

api = None


def config():
    for k, v in settings.items():
        os.environ[k] = v

    global api
    api = wandb.Api()


def download_wandb_file(runid: str, path) -> Union[str, Path]:
    if api is None:
        config()
    run = api.run(runid)
    file = run.file(path).download(f"{wandb_dir}/download-{runid}", replace=True)
    print(f"Downloaded file at wandb://{runid}/{path}")
    return file.name


def download_wandb_files(runid: str, path) -> Union[str, Path]:
    if api is None:
        config()
    run = api.run(runid)
    files = run.files()
    save_dir = f"{wandb_dir}/download-{runid}"
    for file in files:
        if file.name.startswith(path):
            file.download(save_dir, replace=True)
            print(f"Downloaded file at wandb://{runid}/{file.name}")
    return f"{save_dir}/{path}"
