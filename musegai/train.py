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


def train(model, images, rois, outdir, *, labels=None, tag=None, split_axis=None, build_image=True, tempdir=None):
    """train new model on provided datasets

    Args
        model (str): model name
        images: sequence of tuples of files/images
        rois: sequence of files/labelmaps
        labels: ITK label object/file
        tag: model tag (eg. version)
        split_axis: split images into left/right parts
        outdir: output directory for the model files
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
    imagename = "training_{num:03d}_{channel:04d}.nii.gz"
    roiname = "training_{num:03d}.nii.gz"

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
    #with tempfile.TemporaryDirectory(dir=tempdir) as tmp:
    tmp = pathlib.Path(tempdir or 'tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(exist_ok=True, parents=True)

    if 1:

        LOGGER.info("Setup temporary directory")
        # create folder structure
        root = pathlib.Path(tmp)
        (root / "nnUNet_raw").mkdir()
        (root / "nnUNet_results").mkdir()
        (root / "nnUNet_preprocessed").mkdir()
        data_dir=root/"nnUNet_raw/Dataset001"
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
                labelsA, labelsB = labelmap.split(split_axis)
                keepA, keepB = np.any(labelsA > 0), np.any(labelsB > 0)

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
                io.save(data_dir / labeldir / roiname.format(num=num, side="X"), labelmap)
                for channel in range(nchannel):
                    image = io.load(images[index][channel])
                    io.save(data_dir / imagedir / imagename.format(num=num, side="X", channel=channel), image)
                num += 1

        LOGGER.info(f"Done copying training data (num. training: {num})")

        # metadata
        channel_names = {f"{i}": f"mag{i:02d}" for i in range(nchannel)}
        label_names = dict(zip(labels.descriptions, labels.indices))
        label_names['background']=0

        # store JSON metadata
        meta = {
            "channel_names": channel_names,
            "labels": label_names,
            "numTraining": num,
            "file_ending": ".nii.gz",
            "overwrite_image_reader_writer": "SimpleITKIO",
        }
        with open(data_dir/ "dataset.json", "w+") as fp:
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
        io.save_labels(outdir / "labels.txt", labels)

        if build_image:
            # build inference docker
            ...
