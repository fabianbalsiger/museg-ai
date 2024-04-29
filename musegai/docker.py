import docker

"""
TODO:
- create one image per model
- use consistent tags (or simply use `:latest`)
"""

# repository
REPOSITORY = ...
TRAIN_IMAGE = 'fabianbalsiger/museg-train:v1.0.0'


def list_models():
    """list existing models"""
    # temp
    return [
        f"fabianbalsiger/museg:thigh-model3",
    ]
    # client = docker.from_env()
    # images = client.images.search(REPOSITORY)
    # names = [im['name'] for im in images]
    # return names


def run_inference(model, dirname):
    """Run inference"""
    client = docker.from_env()
    image = _get_image(model)
    _pull_image(image)
    print(f"Running model '{model}' (`{image}`)")
    client.containers.run(
        image,
        remove=True,
        device_requests=[docker.types.DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])],
        volumes={dirname: {"bind": "/data", "mode": "rw"}},
    )


def check_training():
    """check if training image available"""
    # TODO
    return True

import pathlib
def run_training(model, indir):
    """Run training"""
    # TODO
    indir = pathlib.Path(indir).resolve()
    client = docker.from_env()
    image = TRAIN_IMAGE
    _pull_image(image)
    print(f"Training model '{model}'")
    container=client.containers.run(
        image,
        "001 3d_fullres 2",
        remove=False,
        # auto_remove=False,
        device_requests=[docker.types.DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])],
        volumes={indir: {"bind": "/data", "mode": "rw"}},
        # volumes={
        #     indir / 'nnUNet_raw': {"bind": "/nnUNet_raw", "mode": "rw"},
        #     indir / 'nnUNet_results': {"bind": "/nnUNet_results", "mode": "rw"},
        #     indir / 'nnUNet_preprocessed': {"bind": "/nnUNet_preprocessed", "mode": "rw"},
        # },
        # environment={"nnUNet_raw":"/data/nnUNet_raw",
        #              "nnUNet_results":"/data/nnUNet_results",
        #              "nnUNet_preprocessed":"/data/nnUNet_preprocessed"},
        detach=True
        
    )

    #affichage des logs du container pour debuggage:
    for line in container.logs(stream=True,follow=True):
        print(line.decode('utf-8').strip())



# private


def _get_image(model):
    """Get Docker image name."""
    # TODO
    return f"fabianbalsiger/museg:{model}"


def _pull_image(image):
    """Pull a Docker image if not exists."""
    client = docker.from_env()
    if not client.images.list(name=image):
        print(f"Pulling image `{image}`, this may take a while...")
        client.images.pull(image)
