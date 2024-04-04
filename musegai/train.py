import pathlib
import json
import logging
import tempfile
import shutil

import numpy as np

from . import io, docker

LOGGER = logging.getLogger(__name__)


"""TODO
- make label values contiguous?
- check output model files
- build docker

"""


def train(model, images, rois, labels, outdir, *, tag=None, split_axis=None, build_image=True, tempdir=None):
    """train new model on provided datasets

    Args
        model (str): model name
        images: sequence of tuples of files/images
        rois: sequence of files/labelmaps
        labels: ITK label object/file
        tempdir: temporary training directory
    Return
        nnU-net model file
        Docker image
    """
    if not docker.check_training():
        raise RuntimeError("Training image not available")

    # names
    imagedir = "imagesTr"
    labeldir = "labelsTr"
    imagename = "training_{index:03d}_{side}_{channel:04d}.nii.gz"
    roiname = "training_{index:03d}_{side}.nii.gz"

    # output model directory
    outdir = pathlib.Path(outdir)
    if outdir.exists():
        raise FileExistsError(outdir)

    nimage = len(images)
    if nimage != len(rois):
        raise ValueError(f"Number of images and rois do not match")
    if not isinstance(images[0], (tuple, list)):
        images = [(im,) for im in images]

    nchannel = len(images[0])
    if {len(im) for im in images} != {nchannel}:
        raise ValueError(f"Number of channels is not constant")

    # check label file
    if labels is not None:
        labels = io.load_labels(labels)

    LOGGER.info("Start training (num. images: {nimage}, num. channels: {nchannel})")

    labelset = None
    with tempfile.TemporaryDirectory(dir=tempdir) as tmp:
        LOGGER.info("Setup temporary directory")
        # create folder structure
        root = pathlib.Path(tmp)
        (root / imagedir).mkdir()
        (root / labeldir).mkdir()

        # check and copy each volume
        num = 0
        for index in range(nimage):
            LOGGER.info(f"Loading dataset {index + 1}/{nimage}")

            # load labelmap
            labelmap = io.load(rois[index])

            # get label values
            _labelset = np.unique(labelmap)

            # check labelset
            if labels and (not set(_labelset) <= set(labels.indices)):
                raise ValueError(f"Labels object do not contain all label values")

            # make label values contiguous?
            # _labelset, labelmap = np.unique(labelmap, return_index=True)

            if labelset is None:
                labelset = set(_labelset)
            elif labelset != set(_labelset):
                raise ValueError(f"Inconsistent number of label values")

            if split_axis is not None:
                # split into halves (eg. left and right sides)
                labelsA, labelsB = labelmap.split(split_axis)
                keepA, keepB = np.any(labelsA > 0), np.any(labelsB > 0)

                # store labelmaps
                if keepA:
                    io.save(root / labeldir / roiname.format(index=index, side="A"), labelsA)
                    num += 1
                if keepB:
                    io.save(root / labeldir / roiname.format(index=index, side="B"), labelsB)
                    num += 1

                # store channels
                for channel in range(nchannel):
                    image = io.load(images[index][channel])
                    imageA, imageB = io.split(image, split_axis)
                    if keepA:
                        io.save(root / imagedir / imagename.format(index=index, side="A", channel=channel), imageA)
                    if keepB:
                        io.save(root / imagedir / imagename.format(index=index, side="B", channel=channel), imageB)

            else:
                # do not split
                io.save(root / labeldir / roiname.format(index=index, side="X"), labelmap)
                for channel in range(nchannel):
                    image = io.load(images[index][channel])
                    io.save(root / imagedir / imagename.format(index=index, side="X", channel=channel), image)
                num += 1

        LOGGER.info(f"Done copying training data (num. training: {num})")

        # label names
        label_names = {f"label_{l}": i for i, l in enumerate(labelset)}

        # store JSON metadata
        meta = {
            "channel_names": ["zscore"] * nchannel,
            "labels": label_names,
            "numTraining": num,
            "file_ending": ".nii.gz",
        }
        with open(root / "datasets.json", "w+") as fp:
            json.dump(meta, fp)

        # run nnU-net training
        breakpoint()
        LOGGER.info(f"Run nnU-net training")
        docker.run_training(model, tmp)

        # store model files
        outdir.mkdir(parents=True, exist_ok=True)
        # TO FIX
        model_data = tmp / "nnUNet_trained_models/nnUNet/3d_fullres/Task503_MuscleThigh/"
        for file in model_data.glob("*"):
            shutil.copyfile(file, outdir / file.name)

        # store label file
        if labels is not None:
            io.save_labels(outdir / "labels.txt", labels)

        if build_image:
            # build inference docker
            ...
