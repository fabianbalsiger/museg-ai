import pathlib
import json
import logging
import tempfile
import shutil
import docker
import numpy as np

from . import dockerutils, io

LOGGER = logging.getLogger(__name__)


"""TODO
- make label values contiguous?
- check output model files
- build docker

"""


def train(model, images, rois, outdir, *, labels=None, split_axis=None, train_model=True, dockerfile=True):
    """train new model on provided datasets

    Args
        model (str): model name
        images: sequence of tuples of files/images
        rois: sequence of files/labelmaps
        labels: ITK label object/file
        split_axis: split images into left/right parts
        outdir: output directory for the model files
        tempdir: temporary training directory
    Return
        nnU-net model file
        Docker image
    """
    if not dockerutils.check_training():
        raise RuntimeError("Training image not available")

    # names
    imagedir = "imagesTr"
    labeldir = "labelsTr"
    imagename = "training_{num:03d}_{channel:04d}.nii.gz"
    roiname = "training_{num:03d}.nii.gz"

    # output model directory
    outdir = pathlib.Path(outdir)

    nimage = len(images)
    if nimage != len(rois):
        raise ValueError(f"Number of images and rois do not match")
    if not isinstance(images[0], (tuple, list)):
        images = [(im,) for im in images]

    nchannel = len(images[0])
    if {len(im) for im in images} != {nchannel}:
        raise ValueError(f"Number of channels is not constant")

    # check label file
    labelremap = None
    if labels is not None:
        labels = io.load_labels(labels)
        if split_axis is not None:
            # remove side suffix in label descriptions
            splits = [descr.rsplit("_", 1) for descr in labels.descriptions]
            labels.descriptions = [split[0] if split[-1].upper() in ["L", "R"] else descr for split, descr in zip(splits, labels.descriptions)]
        # remap index values
        uniquelabels = {name: index for index, name in zip(labels.indices[::-1], labels.descriptions[::-1])}
        labelremap = np.array([uniquelabels[name] for name in labels.descriptions])

    if train_model:
        LOGGER.info("Start training (num. images: {nimage}, num. channels: {nchannel})")

        # create folder structure
        root = pathlib.Path(outdir)
        root.mkdir(parents=True)
        (root / "nnUNet_raw").mkdir()
        (root / "nnUNet_results").mkdir()
        (root / "nnUNet_preprocessed").mkdir()
        data_dir = root / "nnUNet_raw/Dataset001"
        data_dir.mkdir()
        (data_dir / imagedir).mkdir()
        (data_dir / labeldir).mkdir()

        # check and copy each volume
        num = 0
        labelset = None
        for index in range(nimage):
            LOGGER.info(f"Loading dataset {index + 1}/{nimage}")

            # load labelmap
            labelmap = io.load(rois[index])

            if labelremap is not None:
                labelmap.array = labelremap[labelmap.array]

            # make label values contiguous
            _labelset, array = np.unique(labelmap, return_inverse=True)
            labelmap.array = array.reshape(labelmap.shape)

            # setup labels
            if labelset is None:
                labelset = set(_labelset)
                if labels is None:
                    labels = io.init_labels(len(labelset))
                else:
                    # check label indices
                    if not labelset <= set(labels.indices):
                        raise ValueError(f"Labels object does not contain all label values")
                    # reindex labels
                    labels = labels.subset(labelset, reindex=True)
            elif labelset != set(_labelset):
                raise ValueError(f"Inconsistent label values in dataset {index}: {labelset} != {_labelset}")

            if split_axis is not None:
                # split into halves (eg. left and right sides)
                labelsA, labelsB = io.split(labelmap, split_axis)
                keepA, keepB = np.any(labelsA.array > 0), np.any(labelsB.array > 0)

                if keepA:
                    io.save(data_dir / labeldir / roiname.format(num=num), labelsA)
                    for channel in range(nchannel):
                        image = io.load(images[index][channel])
                        imageA, _ = io.split(image, split_axis)
                        io.save(data_dir / imagedir / imagename.format(num=num, channel=channel), imageA)
                    num += 1

                if keepB:
                    io.save(data_dir / labeldir / roiname.format(num=num), labelsB)
                    for channel in range(nchannel):
                        image = io.load(images[index][channel])
                        _, imageB = io.split(image, split_axis)
                        io.save(data_dir / imagedir / imagename.format(num=num, channel=channel), imageB)
                    num += 1

            else:
                # do not split
                io.save(data_dir / labeldir / roiname.format(num=num), labelmap)
                for channel in range(nchannel):
                    image = io.load(images[index][channel])
                    io.save(data_dir / imagedir / imagename.format(num=num, channel=channel), image)
                num += 1

        LOGGER.info(f"Done copying training data (num. training: {num})")

        # metadata
        channel_names = {f"{i}": f"mag{i:02d}" for i in range(nchannel)}
        # force label 0 at 'background'
        labels.descriptions[0] = "background"
        label_names = {labels[i]: i for i in labels}

        # store JSON metadata
        meta = {
            "channel_names": channel_names,
            "labels": label_names,
            "numTraining": num,
            "file_ending": ".nii.gz",
            "overwrite_image_reader_writer": "SimpleITKIO",
        }
        with open(data_dir / "dataset.json", "w+") as fp:
            json.dump(meta, fp)

        # run nnU-net training
        LOGGER.info(f"Run nnU-net training")

        dockerutils.run_training(model, outdir)

        # store model files
        outdir.mkdir(parents=True, exist_ok=True)

        # store label file
        io.save_labels(outdir / "labels.txt", labels)

    if dockerfile:
        LOGGER.info(f"\nBuild docker image for model {model}")
        dockerutils.build_inference(model, outdir, nchannel)
