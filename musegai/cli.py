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


@cli.command(context_settings={"show_default": True})
@click.argument("images", nargs=-1)
@click.option("-d", "--dest", type=click.Path(), help="Output directory.")
@click.option("--filename", default="roi", help="Segmentation filename.")
@click.option("-f", "--format", default=".nii.gz", type=click.Choice([".nii.gz", ".mha", ".mhd", ".hdr"]))
@click.option("--model", default="thigh-model3", help="Specify the segmentation model.")
@click.option("--side", default="LR", type=click.Choice(["L", "R", "LR", "NA"]), help="Limb's side(s) in image")
@click.option("-v", "--verbose", is_flag=True, help="Show more information")
@click.option("--tempdir", type=click.Path(exists=True), help="Location for temporary files.")
@click.option("-r", "--root", type=click.Path(exists=True), help="Root directory for training data.")
@click.option("--overwrite", type=bool, default=False, help="specify if you want to overwrite already existing files in output dir")
def infer(images, dest, filename, format, model, side, tempdir, verbose, overwrite, root):
    """Automatic muscle segmentation command line tool.

    \b
    images can be:
        - (nothing): show available segmentation models
        - a file pattern of images to segment (if multiple channels, images must be numbered)
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    if not images:
        models = api.list_models()
        # no argument: list available models
        click.echo("Available segmentation models:")
        for available_model in models:
            click.echo(f"\t{available_model}")
        sys.exit(0)
    elif len(images) > 1:
        click.echo(f"Invalid number of arguments")
        for ims in images:
            click.echo(f'\t{", ".join(map(str, ims))}')
        sys.exit(1)

    images = images[0]

    # check if the number of channel is consistent
    model_info = api.get_model_info(model)
    nchannel = int(model_info["nchannel"])

    # find images
    if pathlib.Path(images).is_absolute():
        # assume a directory
        image_files = sorted(pathlib.Path(images).rglob("*"))
    else:
        root = pathlib.Path(root) if root else pathlib.Path(".")
        image_files = sorted(root.rglob(images))

    if not image_files:
        click.echo(f"No image file found, check expression: {images}")
        sys.exit(1)

    regex = re.compile(r"(.+?)(\d*)\.([\.\w]+)$")
    images = {}
    for file in image_files:
        match = regex.match(str(file))
        if not match or not api.is_image(file):
            continue
        common, index, ext = match.groups()
        images.setdefault(common, []).append(file)

    images = [tuple(sorted(images[im]))[:nchannel] for im in sorted(images)]

    # destination
    dest = pathlib.Path(root if dest is None else dest)
    dest.mkdir(exist_ok=True, parents=True)
    destfiles = {name: (dest / name[0].parent / filename).with_suffix(format) for name in images}

    inputs = images
    outputs = list(destfiles.values())

    click.echo(f"Found {len(images)} image to segment (num. channels: {nchannel}):")
    for i in range(len(inputs)):
        click.echo(f"({i+1})")
        for j in range(nchannel):
            click.echo(f"\tchan1:  {images[i][j]}")
        click.echo(f"\tlabels:  {outputs[i]}")

    # dealing whith already existent files in output directory
    if not overwrite:
        for name in list(destfiles):
            if destfiles[name].is_file():
                click.echo(f"Output file already exists: {destfiles[name]}, skipping")
                images.remove(name)
                destfiles.pop(name)

    if not images:
        click.echo("Nothing to do.")
        sys.exit(0)

    # side
    if side == "NA":
        side = None

    # segment images
    click.echo(f"Segmenting {len(images)} volume(s)...")
    api.run_model(model, inputs, outputs, side=side, tempdir=tempdir)

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
@click.option("--split", is_flag=True, help="Split datasets into left and right parts")
@click.option("-v", "--verbose", is_flag=True)
@click.option("--folds", help="specify fold numbers to train as tuple")
def train(model, images, rois, train, dockerfile, nchannel, labelfile, root, dest, split, verbose, folds, preprocess):
    """Create new segmentation model using training images and rois"""
    if verbose:
        logging.basicConfig(level=logging.INFO)

    # output dir
    if not dest:
        dest = pathlib.Path(".") / model.replace(":", "_")
    else:
        dest = pathlib.Path(dest)

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
    roi_dates = list(roi_dates.values())

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
            click.echo(f"\tchan. {j + 1:02d}:  {images[i][j]}")
        click.echo(f"\tlabels:    {rois[i]}")
        if len(roi_dates[i]) > 1:
            click.echo(f'\t(latest among: {", ".join(roi_dates[i])})')
    ans = click.confirm("Are all images/rois correctly matched?", abort=True)

    # train model
    split_axis = None if not split else 0
    api.train_model(model, images, rois, labelfile, dest, split_axis=split_axis, train_model=train, make_dockerfile=dockerfile, folds=folds, preprocess=preprocess)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
