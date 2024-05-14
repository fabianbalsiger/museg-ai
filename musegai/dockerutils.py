import pathlib
import shutil
import docker
from . import docker_template

"""
TODO:
- create one image per model
- use consistent tags (or simply use `:latest`)
"""

# repository
REPOSITORY = ...
TRAIN_IMAGE = "museg-train:v1.1.0"


def list_models(local=True):
    """list existing models"""
    models = {}
    if local:
        client = docker.from_env()
        for image in client.images.list():
            if "museg" in image.tags[0]:
                models[image.tags[0]] = image.labels
    return models

    # return [
    #     f"fabianbalsiger/museg:thigh-model3",
    # ]
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
        ipc_mode="host",
        device_requests=[docker.types.DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])],
        volumes={dirname: {"bind": "/data", "mode": "rw"}},
    )
    

def check_training():
    """check if training image available"""
    # TODO
    return True


def run_training(model, dirname):
    """Run training"""
    dirname = str(pathlib.Path(dirname).resolve())
    client = docker.from_env()
    image = TRAIN_IMAGE
    _pull_image(image)

    print(f"Training model '{model}'")

    def run(entrypoint, cmd):
        container = client.containers.run(
            image,
            cmd,
            entrypoint=entrypoint,
            remove=True,
            detach=True,
            ipc_mode="host",
            device_requests=[docker.types.DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])],
            volumes={dirname: {"bind": "/nnunet", "mode": "rw"}},
        )
        # print logs
        for line in container.logs(stream=True, follow=True):
            print(line.decode("utf-8").strip())
        return container.status

    # preprocess
    status = run("nnUNetv2_plan_and_preprocess", ["-d", "001", "-c", "3d_fullres", "--verify_dataset_integrity"])

    # train
    status = run("nnUNetv2_train", ["001", "3d_fullres", "0"])
    status = run("nnUNetv2_train", ["001", "3d_fullres", "1"])
    status = run("nnUNetv2_train", ["001", "3d_fullres", "2"])
    status = run("nnUNetv2_train", ["001", "3d_fullres", "3"])
    status = run("nnUNetv2_train", ["001", "3d_fullres", "4"])


def get_ressources():
    """get the path to the ressources folder"""
    here = pathlib.Path(__file__).parent
    return here.parent / "docker" / "training"


def build_inference(model, tag, dirname):
    """build inference docker"""
    client = docker.from_env()

    # get docker template
    dockerfile = docker_template.make_docker(model, dirname, folds=(0, 1, 2, 3, 4))
    # dockerfile = docker_template.make_docker(model, dirname, folds=(0,))

    # get the requirement file
    ressources_dir = get_ressources()
    shutil.copy((ressources_dir / "requirements.txt"), dirname)

    # dockerfile writing
    with open(dirname / "Dockerfile", "w") as fp:
        fp.write(dockerfile)

    # build image
    print("Run the following command to build the model's docker:")
    print("docker build <outdir> --tag <model>:<tag>")
    # image, logs = client.images.build(path=str(dirname), tag=tag, quiet=False, forcerm=True, rm=True)
    # for chunk in logs:
    #     if not "stream" in chunk:
    #         continue
    #     for line in chunk["stream"].splitlines():
    #         print(line)


# private


def _get_image(model):
    """Get Docker image name."""
    if ":" in model: #if arg is a valid docker name 
        return model
    # TODO
    #return default model if input is not a valid docker name
    else: 
        return f"fabianbalsiger/museg:{model}"


def _pull_image(image):
    """Pull a Docker image if not exists."""
    client = docker.from_env()
    if not client.images.list(name=image):
        print(f"Pulling image `{image}`, this may take a while...")
        client.images.pull(image)
