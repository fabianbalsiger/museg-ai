import pathlib
import shutil

def get_ressources():
    """get the path to the ressources folder"""
    here = pathlib.Path(__file__).parent
    return here.parent / "docker" /"training"

def make_docker(title, outdir, f=(0, 1, 2, 3, 4), d="001"):
    """create the dockerfile from the template below"""
    ressources_dir = get_ressources()
    # get the requirement file
    shutil.copy((ressources_dir/ "requirements.txt"),outdir)
    # dealing with folder link with the fold number in crossvalidation
    checkpoints_files = ""
    for fold_nbr in f:
        # checkpoints_files = (
        #     checkpoints_files
        #     + f"COPY ./nnUNet_results/Dataset001/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{fold_nbr}/checkpoint_final.pth /nnUNet_results/Dataset001/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{fold_nbr}/checkpoint_final.pth\n"
        # )
        checkpoints_files = (
            checkpoints_files
            + f"COPY ./nnUNet_results/Dataset001/nnUNetTrainer_1epoch__nnUNetPlans__3d_fullres/fold_{fold_nbr}/checkpoint_final.pth /nnUNet_results/Dataset001/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{fold_nbr}/checkpoint_final.pth\n"
        )
    # putting f in good shape to be processed by CLI :
    str_out=''
    for nbr in f :
        str_out=str_out+"\""+ str(nbr)+ "\""+","
    f=str_out[-1]
    dockerfile_content = DOCKER_FILE.format(title=title, ressources_dir=ressources_dir, outdir=outdir, f=f, d=d, checkpoints_files=checkpoints_files)

    # dockerfile writing
    with open(outdir / "Dockerfile", "w") as dockerfile:
        dockerfile.write(dockerfile_content)

    print("Dockerfile successfully created!")
    if (get_ressources() / 'requirements.txt').exists():
        print('fichier requirement trouvé')


DOCKER_FILE = """FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04
LABEL application={title}
LABEL author="Fabian Balsiger and Pierre-Yves Baudin"

ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /

# SYSTEM
RUN apt update --yes --quiet && apt install --yes --quiet --no-install-recommends software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt install --yes --quiet --no-install-recommends \
    python3.10 libpython3.10-dev python3.10 python3.10-distutils build-essential curl \
&& rm -rf /var/lib/apt/lists/*

# Switch default Python version to 3.10
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
update-alternatives --install /usr/bin/python python /usr/bin/python3.10 2 && \
update-alternatives --set python /usr/bin/python3.10 && \
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2 && \
update-alternatives --set python3 /usr/bin/python3.10

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py

# Copy necessary files to the working directory
COPY ./requirements.txt .

# Copy the model to the working directory
{checkpoints_files}
COPY ./nnUNet_results/Dataset001/nnUNetTrainer_1epoch__nnUNetPlans__3d_fullres/dataset.json /nnUNet_results/Dataset001/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json
COPY ./nnUNet_results/Dataset001/nnUNetTrainer_1epoch__nnUNetPlans__3d_fullres/dataset_fingerprint.json /nnUNet_results/Dataset001/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset_fingerprint.json
COPY ./nnUNet_results/Dataset001/nnUNetTrainer_1epoch__nnUNetPlans__3d_fullres/plans.json /nnUNet_results/Dataset001/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json
# COPY ./nnUNet_results/Dataset001/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json /nnUNet_results/Dataset001/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json
# COPY ./nnUNet_results/Dataset001/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset_fingerprint.json /nnUNet_results/Dataset001/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset_fingerprint.json
# COPY ./nnUNet_results/Dataset001/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json /nnUNet_results/Dataset001/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Set nnU-Net environment variable
ENV nnUNet_results="/nnUNet_results"
ENV nnUNet_raw="/nnUNet_raw"
ENV nnUNet_preprocessed = "/nnUNet_preprocessed"

COPY ./labels.txt ./labels.txt

CMD ["-i", "/data/in", "-o", "/data/out"]
ENTRYPOINT ["nnUNetv2_predict", "-c", "3d_fullres", "-d", \"{d}\", "-f", {f}, "--save_probabilities"]
"""
