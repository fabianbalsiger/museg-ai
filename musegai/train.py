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


def train(
    model,
    images,
    rois,
    labels,
    outdir,
    *,
    split_axis=None,
    train_model=True,
    make_dockerfile=True,
    folds=(0, 1, 2, 3, 4),
    nepoch=250,
    preprocess=True,
    random_pruning=True,
    continue_training=False,
):
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
    labels = io.load_labels(labels)
    if split_axis is not None:
        # remove side suffix in label descriptions
        splits = [descr.rsplit("_", 1) for descr in labels.descriptions]
        labels.descriptions = [split[0] if split[-1].upper() in ["L", "R"] else descr for split, descr in zip(splits, labels.descriptions)]
    # set 0 as background
    labels.descriptions[0] = "background"
    # add ignore label
    labels.append("ignore", color=(255, 0, 0), visibility=0)
    ignore_label = labels["ignore"]
    # remap index values
    uniquelabels = set(labels.descriptions)
    labelset = set(labels[name] for name in uniquelabels)
    labelremap = -1 * np.ones(max(labels.indices) + 1, dtype=int)
    labelremap[np.array(list(labelset), dtype=int)] = np.arange(len(uniquelabels))
    # reindex labels
    labels = labels.subset(labelset)

    # random state
    rstate = np.random.RandomState(0)

    if train_model and preprocess:
        LOGGER.info(f"Start training (num. images: {nimage}, num. channels: {nchannel})")

        # create folder structure
        root = pathlib.Path(outdir)
        root.mkdir(parents=True,exist_ok=True)
        (root / "nnUNet_raw").mkdir(exist_ok=True)
        (root / "nnUNet_results").mkdir(exist_ok=True)
        (root / "nnUNet_preprocessed").mkdir(exist_ok=True)
        data_dir = root / "nnUNet_raw/Dataset001"
        data_dir.mkdir(exist_ok=True)
        (data_dir / imagedir).mkdir(exist_ok=True)
        (data_dir / labeldir).mkdir(exist_ok=True)

        # check and copy each volume
        num = 0
        # labelset = None
        for index in range(nimage):
            LOGGER.info(f"Loading dataset {index + 1}/{nimage}")

            # load labelmap
            labelmap = io.load(rois[index])

            # add ignore label
            empty_slices = np.all(labelmap.array == 0, axis=(0, 1))
            labelmap.array[:, :, empty_slices] = ignore_label

            nprune = 0
            if random_pruning:
                # remove slices at the bottom
                nprune = rstate.randint(0, np.argmin(empty_slices) + 1)
                if nprune:
                    LOGGER.info(f"Pruning {nprune} slices at the bottom of the volume")
                    labelmap.array = labelmap.array[:, :, nprune:]

            # check labels
            _labelset = np.unique(labelmap)
            if not set(_labelset) <= labelset:
                raise ValueError(f"Inconsistent label values in dataset {index + 1}: {_labelset} not included in {labelset}.")
            elif set(_labelset) != labelset:
                diff = [labels[i] for i in (set(_labelset) - labelset)]
                print(f"Warning, in dataset {index + 1}, some labels are missing: {diff}")

            # remap labelmap (remove duplicate label names, make labels consecutive)
            labelmap.array = labelremap[labelmap.array]

            if split_axis is not None:
                # split into halves (eg. left and right sides)
                labelsA, labelsB = io.split(labelmap, split_axis)
                keepA, keepB = np.any(labelsA.array > 0), np.any(labelsB.array > 0)

                if keepA:
                    io.save(data_dir / labeldir / roiname.format(num=num), labelsA)
                    for channel in range(nchannel):
                        image = io.load(images[index][channel])
                        image.array = image.array[..., nprune:]
                        imageA, _ = io.split(image, split_axis)
                        io.save(data_dir / imagedir / imagename.format(num=num, channel=channel), imageA)
                    num += 1

                if keepB:
                    io.save(data_dir / labeldir / roiname.format(num=num), labelsB)
                    for channel in range(nchannel):
                        image = io.load(images[index][channel])
                        image.array = image.array[..., nprune:]
                        _, imageB = io.split(image, split_axis)
                        io.save(data_dir / imagedir / imagename.format(num=num, channel=channel), imageB)
                    num += 1

            else:
                # do not split
                io.save(data_dir / labeldir / roiname.format(num=num), labelmap)
                for channel in range(nchannel):
                    image = io.load(images[index][channel])
                    image.array = image.array[..., nprune:]
                    io.save(data_dir / imagedir / imagename.format(num=num, channel=channel), image)
                num += 1
            #adding click channels
            def add_click_chan(image,labels):
                """image is one single image (case) with all its channels"""
                #labels is a label type 
                nlabels=len(labels)
                label_list=[k for k in labels.__iter__()]
                if 'ignore' in labels.__getitem__(label_list[-1]):
                     nlabels= nlabels-1
                click_chan=io.load(image[0])
                for k in range(nlabels):
                    click_chan.array = 0 * click_chan.array
                    io.save(data_dir/imagedir/imagename.format(num=num-1, channel=nchannel+k), click_chan)
            
            interactive=True
            if interactive:
                add_click_chan(images[index],labels)
         # metadata
        channel_names = {f"{i}": f"mag{i:02d}" for i in range(nchannel)}
        label_names = {labels[i]: i for i in labels}

       
           
        if interactive:
            if 'ignore' in labels.__getitem__(labels.indices[-1]):
                label_count=len(labels.indices)-1
            else:
                label_count=len(labels.indices)
            for k in range(label_count): #addding metadata for click channels
                channel_names[f"{nchannel+k}"]="noNorm"

        

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

        # store label file
        if "ignore" in labels.descriptions:
            labels = labels.remove("ignore")
        io.save_labels(outdir / "labels.txt", labels)

        LOGGER.info(f"Done copying training data (num. training: {num})")

    if train_model:
        # run nnU-net training
        LOGGER.info(f"Run nnU-net training")
        dockerutils.run_training(model, outdir, nepoch=nepoch, folds=folds, preprocess=preprocess, continue_training=continue_training)

    if make_dockerfile:
        LOGGER.info(f"\nGenerate dockerfile for model {model}")
        # list folds
        fold_dirs = list((outdir / "nnUNet_results").rglob("fold_*/checkpoint_final.pth"))
        folds = [int(dirname.parent.name.split("_")[1]) for dirname in fold_dirs]
        # make dockerfile
        dockerutils.make_dockerfile(model, outdir, nchannel, folds=folds, nepoch=nepoch)
