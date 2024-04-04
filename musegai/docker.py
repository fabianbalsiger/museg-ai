import docker

"""
TODO:
- create one image per model
- use consistent tags (or simply use `:latest`)
"""

# repository
REPOSITORY = ...
TRAIN_IMAGE = ...


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


def run_training(model, indir):
    """Run training"""
    # TODO
    client = docker.from_env()
    image = _get_image(TRAIN_IMAGE)
    _pull_image(image)
    print(f"Training model '{model}'")
    client.containers.run(
        image,
        remove=True,
        device_requests=[docker.types.DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])],
        volumes={indir.parent: {"bind": "/data", "mode": "rw"}},
    )


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
