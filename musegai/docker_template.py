
def make_docker(title, outdir,f=(0,1,2,3,4),d="001"):
    checkpoints_files=""
    for fold_nbr in f: 
        checkpoints_files=checkpoints_files+f"COPY {outdir}/nnUNet_results/Dataset001/nnUNetTrainer_nnUNetPlans_3D_fullres/fold_{fold_nbr}/checkpoint_final.pth ./nnUNet_results/Dataset001/nnUNetTrainer_nnUNetPlans_3D_fullres/fold_{fold_nbr}/checkpoint_final.pth\n"
    dockerfile_content = f'''FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04
LABEL application={title}
LABEL author="Fabian Balsiger and Pierre-Yves Baudin"

ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /

# SYSTEM
RUN apt update --yes --quiet && apt install --yes --quiet --no-install-recommends software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt install --yes --quiet --no-install-recommends \
    python3.10 python3.10-distutils build-essential curl \
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
COPY ./requirements_train.txt .

# Copy the model to the working directory
{checkpoints_files}
COPY {outdir}/nnUNet_results/Dataset001/nnUNetTrainer_nnUNetPlans_3D_fullres/dataset.json ./nnUNet_results/Dataset001/nnUNetTrainer_nnUNetPlans_3D_fullres/dataset.json
COPY {outdir}/nnUNet_results/Dataset001/nnUNetTrainer_nnUNetPlans_3D_fullres/dataset_fingerprint.json ./nnUNet_results/Dataset001/nnUNetTrainer_nnUNetPlans_3D_fullres/dataset_fingerprint.json
COPY {outdir}/nnUNet_results/Dataset001/nnUNetTrainer_nnUNetPlans_3D_fullres/plans.json ./nnUNet_results/Dataset001/nnUNetTrainer_nnUNetPlans_3D_fullres/plans.json

# Install requirements
RUN pip install --no-cache-dir -r requirements_train.txt

# Copy custom trainer and set nnU-Net environment variable
ENV RESULTS_FOLDER="./nnUNet_trained_models"

COPY {outdir}/labels.txt ./labels.txt

CMD ["-i", "./data/in", "-o", "./data/out"]
ENTRYPOINT ["nnUNetv2_predict.py","-c", "3d_fullres", "-p", "nnUNetPlansv2.1", '--save_probabilities', "-d", {d},"f", {f}]
'''

    # Création du dossier docker & écriture du dockerfile
    (outdir / 'infer_docker').mkdir(parents=True, exist_ok=True)
    with open(outdir / 'infer_docker' / 'Dockerfile', 'w') as dockerfile:
        dockerfile.write(dockerfile_content)

    print("Dockerfile successfully created!")
