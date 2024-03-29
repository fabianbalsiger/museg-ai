import pathlib
import shutil
import logging
import tempfile
import numpy as np

from . import io, docker

LOGGER = logging.getLogger(__name__)


def infer(model, images, outputs=None, *, split_axis=None, tempdir=None):
    """ run inference on images
    
    Args
        model (str): model name
    """

    # names
    indir = 'in'
    outdir = 'out'
    imagename = 'im_{index:03d}_{side}_{channel:04d}.nii.gz'
    roiname = 'labels_{index:03d}_{side}.nii.gz'
    
    nimage = len(images)
    if not isinstance(images[0], (tuple, list)):
        images = [(im,) for im in images]

    nchannel = len(images[0])
    if {len(im) for im in images} != {nchannel}:
        raise ValueError(f'Number of channels is not constant')
    
    if outputs and len(set(outputs)) != nimage:
        raise ValueError('Mismatch in the number of inputs and outputs')
    
    labelnames = None
    with tempfile.TemporaryDirectory(dir=tempdir) as tmp:
        LOGGER.info('Setup temporary directory')
        # create folder structure
        root = pathlib.path(tmp)
        (root / indir).mkdir()
        (root / outdir).mkdir()
        
        # check and copy each volume
        num = 0
        for index in range(nimage):
            LOGGER.info(f'Loading dataset {index + 1}/{nimage}') 
            if split_axis is not None:
                # split into halves (eg. left and right sides)
                LOGGER.info(f'Split images into halves')
                for channel in range(nchannel):
                    image = io.load(images[index][channel])
                    imageA, imageB = image.split(split_axis)
                    io.save(root / indir / imagename.format(index=index, side='A', channel=channel), imageA)
                    io.save(root / indir / imagename.format(index=index, side='B', channel=channel), imageB)
                num += 2
            else:
                # do not split
                for channel in range(nchannel):
                    image = io.load(images[index][channel])
                    io.save(root / indir / imagename.format(index=index, side='X', channel=channel), image)
                num += 1
        
        LOGGER.info(f'Done copying data (num. images: {num})')

        # run model
        docker.run_inference(model, root)

        # recover outputs
        labelmaps = []
        for index in range(num):
            if split_axis:
                labelmapA = io.load(root / outdir / roiname.format(index=index, side='A'))
                labelmapB = io.load(root / outdir / roiname.format(index=index, side='B'))
                labelmap = labelmapA.heal(labelmapB, axis=split_axis)
            else:
                labelmap = io.load(root / outdir / roiname.format(index=index, side='X'))

        labelmaps.append(labelmap)

        if labelnames is None:
            # get label names
            labelnames = io.Labels.load(root / outdir / 'labels.txt')


    # return volumes or files?
    if not outputs:
        return labelmaps, labelnames
    
    for filename, labelmap in zip(outputs, labelmaps):
        io.save(filename, labelmap)
    io.Labels.save(outdir / 'labels.txt', labelnames)

    

    
        