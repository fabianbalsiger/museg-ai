"""Command line interface for muscle segmentation."""

from __future__ import annotations
import logging
import pathlib
import re
import sys

import click

from musegai import api


@click.group()
def cli(): ...

@cli.command()
def list():
    """ List available models """
    models = api.list_models()
    # no argument: list available models
    click.echo("Available segmentation models:")
    for available_model in models:
        click.echo(f"\t{available_model}")
    sys.exit(0)


@cli.command(context_settings={"show_default": True})
@click.argument("model")
@click.argument("images", nargs=-1)
@click.option('-o', "--overwrite", is_flag=True, help="Overwrite already existing files.")
@click.option("-r", "--root", type=click.Path(exists=True), help="Input root directory.")
@click.option("-d", "--dest", type=click.Path(), help="Output root directory.")
@click.option("--filename", default="roi", help="Segmentation filename.")
@click.option("--dirname", help="Segmentation parent folder.")
@click.option("-f", "--format", default=".nii.gz", type=click.Choice([".nii.gz", ".mha", ".mhd", ".hdr"]))
@click.option("--side", default="LR", type=click.Choice(["L", "R", "LR", "NA"]), help="Limb's side(s) in image")
@click.option("--tempdir", type=click.Path(exists=True), help="Location for nnUNet temporary files.")
@click.option("-v", "--verbose", is_flag=True, help="Show more information")
def segment(images, dest, dirname, filename, format, model, side, tempdir, verbose, overwrite, root):
    """Automatic muscle segmentation command line tool.

    \b
    IMAGES: file name(s) or pattern(s) of images to segment.
    Multiple channels are assumed either:
    - if image files are numbered
    - if multiple patterns are passed
    
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    # copy inputs to outputs
    copy_inputs = (dirname is not None) or (dest is not None)

    root = pathlib.Path(root) if root else pathlib.Path(".")

    if not images:
        models = api.list_models()
        # no argument: list available models
        click.echo("Available segmentation models:")
        for available_model in models:
            click.echo(f"\t{available_model}")
        sys.exit(0)

    # get nchannel from model metadata
    try:
        model_info = api.get_model_info(model)
    except ValueError as exc:
        click.echo(f'Error: {exc}')
        sys.exit(1)

    if model == "fabianbalsiger/museg:thigh-model3":
        nchannel = 2
    else:
        nchannel = int(model_info["nchannel"])

    if 1 < len(images) != nchannel:
        click.echo(f"Expecting {nchannel} channels, got {len(images)} expression(s).")
        sys.exit(1)

    image_files = []
    for expr in images:
        # find images
        if pathlib.Path(expr).is_absolute():
            # assume a directory
            click.echo(f"\tAbsolute paths are not accepted: {expr}")
        files = sorted(root.rglob(expr))
        image_files.append(files)

    num_files = {expr: len(files) for expr, files in zip(images, image_files)}
    if any(num == 0 for num in num_files):
        click.echo(f"No image files found, check expression(s): {[ex for ex in num_files if num_files[ex] == 0]}")
        sys.exit(1)

    images = {}
    if len(image_files) == 1:
        # numbered channels
        regex = re.compile(r"(.+?)(\d*)\.([\.\w]+)$")
        for file in image_files[0]:
            match = regex.match(str(file))
            if not match or not api.is_image(file):
                continue
            common, index, ext = match.groups()
            images.setdefault(common, []).append(file)
    else:
        # one expression per channel
        regex = re.compile(r"(.+?)\.([\.\w]+)$")
        for files in image_files:
            for file in files:
                match = regex.match(str(file))
                if not match or not api.is_image(file):
                    continue
                common = file.parent
                images.setdefault(common, []).append(file)

    # check number of channels
    names = sorted(images)
    numchan = {name: len(images[name]) for name in names if len(images[name]) != nchannel}
    if numchan:
        click.echo(f"Did not find {nchannel} channel(s) in:")
        for name in numchan:
            click.echo(f'\t{name}: {", ".join(f"[{i + 1}] {file}" for i, file in enumerate(images[name]))}')
        sys.exit(1)

    # destination
    dest = pathlib.Path(root if dest is None else dest)
    dest.mkdir(exist_ok=True, parents=True)
    destloc = {name: images[name][0].relative_to(root).parent for name in names}
    if dirname:
        destloc = {name: loc.parent / dirname for name, loc in destloc.items()}
    labels = {name: (dest / destloc[name] / filename).with_suffix(format) for name in names}

    click.echo(f"Found {len(images)} image to segment (num. channels: {nchannel}):")
    for i, name in enumerate(names):
        click.echo(f"({i+1})")
        for j in range(nchannel):
            click.echo(f"\tchan. {j + 1:02d}: {images[name][j]}")
        click.echo(f"\tlabels:   {labels[name]}")
    click.confirm("Are all images correctly selected?", abort=True)

    # dealing whith already existent files in output directory
    if not overwrite:
        for name in names:
            if labels[name].is_file():
                click.echo(f"Output file already exists: {labels[name]}, skipping")
                images.pop(name)
                labels.pop(name)

    if not images:
        click.echo("Nothing to do.")
        sys.exit(0)

    # side
    if side == "NA":
        side = None

    inputs = [images[name] for name in names]
    outputs = [labels[name] for name in names]

    # segment images
    click.echo(f"Segmenting {len(images)} volume(s)...")
    api.run_model(model, inputs, outputs, side=side, tempdir=tempdir, copy_inputs=copy_inputs)

    click.echo("Done.")


@cli.command(context_settings={"show_default": True})
@click.argument("model")
@click.argument("images")
@click.argument("rois")
@click.option("--labelfile", type=click.Path(exists=True), required=True, help="ITK-Snap label file")
@click.option("--train/--no-train", default=True, help="Train model.")
@click.option("--dockerfile/--no-dockerfile", default=True, help="Make dockerfile.")
@click.option("--preprocess/--no-preprocess", default=True, help="enable or not the preprocessing part")
@click.option("-r", "--root", type=click.Path(exists=True), help="Root directory for training data.")
@click.option("-d", "--dest", type=click.Path(), help="Output directory for model files.")
@click.option("--nchannel", type=int, default=1, help="Expected number of channels")
@click.option("--folds", help="specify fold numbers to train as tuple")
@click.option("--nepoch", type=click.Choice(["1", "10", "100", "250", "1000"]), default="250", help="Number of epochs")
# @click.option("--split", is_flag=True, help="Split datasets into left and right parts")
@click.option("--continue", "continue_training", is_flag=True, help="Continue training.")
@click.option("-v", "--verbose", is_flag=True)
def train(model, images, rois, train, dockerfile, nchannel, labelfile, root, dest, verbose, folds, preprocess, nepoch, continue_training):
    """Create new segmentation model using training images and rois"""
    if verbose:
        logging.basicConfig(level=logging.INFO)
    nepoch = int(nepoch)

    # output dir
    if not dest:
        dest = pathlib.Path(".") / model.replace(":", "_")
    else:
        dest = pathlib.Path(dest.strip())

    if set(pathlib.Path(dest).glob("*")) and train:
        click.confirm(f"Output folder `{dest}` is not empty, do you want to continue?", abort=True)

    if folds is None:
        folds = (0, 1, 2, 3, 4)
    else:
        folds = tuple(map(int, folds.split(",")))

    # find images
    if pathlib.Path(images).is_absolute():
        # assume a directory
        image_files = sorted(pathlib.Path(images).rglob("*"))
        roi_files = sorted(pathlib.Path(rois).rglob("*"))
    else:
        root = pathlib.Path(root) if root else pathlib.Path(".")
        image_files = sorted(root.rglob(images))
        roi_files = sorted(root.rglob(rois))

    if not image_files:
        click.echo(f"No image file found, check expression: {images}")
    if not roi_files:
        click.echo(f"No label file found, check expression: {rois}")

    regex = re.compile(r"(.+?)(\d*)\.([\.\w]+)$")
    images = {}
    for file in image_files:
        match = regex.match(str(file))
        if not match or not api.is_image(file):
            continue
        prefix, index, ext = match.groups()
        images.setdefault(prefix, []).append(file)

    regex = re.compile(r"(.+?)(20\d\d\d\d\d\d)(.+)")
    rois, roi_dates = {}, {}
    for file in roi_files:
        match = regex.match(str(file))
        if not api.is_image(file):
            continue
        elif not match:  # no date
            prefix = str(file)
            date = 0
        else:
            prefix, date, _ = match.groups()
        rois.setdefault(prefix, []).append(file)
        roi_dates.setdefault(prefix, []).append(date)

    images = [tuple(sorted(images[prefix]))[:nchannel] for prefix in sorted(images)]
    rois = [tuple(sorted(rois[prefix]))[-1] for prefix in sorted(rois)]
    roi_dates = tuple(roi_dates.values())

    nimage = len(images)
    if len(rois) != nimage:
        click.echo(f"Error: found {nimage} images and {len(rois)} rois.")
        for ims in images:
            click.echo(f'\t{", ".join(map(str, ims))}')
        for im in rois:
            click.echo(f"\t{im}")
        sys.exit(0)

    if not images:
        click.echo(f"No image files were found")
        sys.exit(0)

    # check num channels
    channels = {len(ims) for ims in images}
    if not channels == {nchannel}:
        click.echo(f"Error: invalid number of channels (must be {nchannel})")
        for ims in images:
            click.echo(f'\t{", ".join(map(str, ims))}')
        sys.exit(0)

    # check training data
    click.echo(f"Found {len(images)} images and matching rois:")
    for i in range(nimage):
        click.echo(f"({i+1})")
        for j in range(nchannel):
            click.echo(f"\tchan. {j + 1:02d}: {images[i][j]}")
        click.echo(f"\tlabels:   {rois[i]}")
        if len(roi_dates[i]) > 1:
            click.echo(f'\t(latest among: {", ".join(roi_dates[i])})')
    ans = click.confirm("Are all images/rois correctly matched?", abort=True)

    # train model
    split_axis = 0 #None if not split else 0
    api.train_model(
        model,
        images,
        rois,
        labelfile,
        dest,
        split_axis=split_axis,
        train_model=train,
        make_dockerfile=dockerfile,
        folds=folds,
        preprocess=preprocess,
        nepoch=nepoch,
        continue_training=continue_training,
    )


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
