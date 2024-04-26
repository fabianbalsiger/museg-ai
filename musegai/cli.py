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
@click.argument("images", type=click.Path(exists=True), nargs=-1)
@click.option("-d", "--dest", type=click.Path(), help="Output directory.")
@click.option("-f", "--format", default=".nii.gz", type=click.Choice([".nii.gz", ".mha", ".mhd", ".hdr"]))
@click.option("--model", default="thigh-model3", help="Specify the segmentation model.")
@click.option("--side", default="LR", type=click.Choice(["L", "R", "LR", "NA"]), help="Limb's side(s) in image")
@click.option("-v", "--verbose", is_flag=True, help="Show more information")
@click.option("--tempdir", type=click.Path(exists=True), help="Location for temporary files.")
def infer(images, dest, format, model, side, tempdir, verbose):
    """Automatic muscle segmentation command line tool.

    \b
    images can be:
        - (nothing): show available segmentation models
        - two matching Dixon images to segment
        - a single directory with numbered pairs of matching Dixon images
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    if not images:
        # no argument: list available models
        click.echo("Available segmentation models:")
        for available_model in api.list_models():
            click.echo(f"\t{available_model}")
        sys.exit(0)

    if (len(images) == 1) and pathlib.Path(images[0]).is_dir():
        # a folder with volume pairs
        root = pathlib.Path(images[0])
        regex = re.compile(r"(.+?)(\d+).[\w.]+$")
        images = {}
        for file in sorted(root.glob("*")):
            match = regex.match(file.name)
            if not match:
                continue
            name, _ = match.groups()
            images.setdefault(name, []).append(file)
            if len(images[name]) > 2:
                click.echo(f"Expecting two volume files with prefix: {name}")
        click.echo(f"Found {len(images)} volume pair(s) to segment:")
        for name in images:
            click.echo(f"\t{name}")

    elif len(images) == 2 and all(pathlib.Path(file).is_file() for file in images):
        # individual files
        root = "."
        images = [pathlib.Path(file) for file in images]
        name = images[0].name
        images = {name: images}
        click.echo(f"Found one volume pair to segment: {name}")

    else:
        # invalid number of arguments
        click.echo("Expecting two volume files or a single directory")
        sys.exit(0)

    # destination
    dest = pathlib.Path(root if dest is None else dest)
    dest.mkdir(exist_ok=True, parents=True)
    destfiles = {name: (dest / name).with_suffix(format) for name in images}
    for name in list(destfiles):
        if destfiles[name].is_file():
            click.echo(f"Output file already exists: {destfiles[name]}, skipping")
            images.pop(name)
            destfiles.pop(name)

    if not images:
        click.echo("Nothing to do.")
        sys.exit(0)

    # side
    if side == "NA":
        side = None

    # segment images
    click.echo(f"Segmenting {len(images)} volume(s)...")
    inputs = list(images.values())
    outputs = list(destfiles.values())
    api.run_model(model, inputs, outputs, side=side, tempdir=tempdir)

    click.echo("Done.")


@cli.command(context_settings={"show_default": True})
@click.argument("model")
@click.argument("images")
@click.argument("rois")
@click.option("--labelfile", type=click.Path(exists=True), help="ITK-Snap label file")
@click.option("--nchannel", type=int, default=2, help='Expected number of channels')
@click.option("-r", "--root", type=click.Path(exists=True), help="root directory.")
@click.option("-d", "--dest", type=click.Path(), help="Output directory.")
@click.option("--split", is_flag=True, help="Split datasets into left and right parts")
@click.option("-v", "--verbose", is_flag=True, help="Show more information")
@click.option("--tempdir", type=click.Path(exists=True), help="Location for temporary files.")
def train(model, images, rois, nchannel, labelfile, root, dest, split, tempdir, verbose):
    """Create new segmentation model using training images and rois"""
    # output dir
    if not dest:
        outdir = pathlib.Path(".") / model
    if set(pathlib.Path(outdir).glob("*")):
        click.echo(f"Output folder `{outdir}` is not empty, exiting.")
        sys.exit(1)

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
        common, index, ext = match.groups()
        images.setdefault(common, []).append(file)

    images = [tuple(sorted(images[im]))[:nchannel] for im in sorted(images)]
    rois = sorted(file for file in roi_files if api.is_image(file))

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
        click.echo(f"Error: invalid number of channels (must be 2)")
        for ims in images:
            click.echo(f'\t{", ".join(map(str, ims))}')
        sys.exit(0)

    # check training data
    click.echo(f"Found {len(images)} images and matching rois:")
    for i in range(nimage):
        click.echo(f"({i+1})")
        click.echo(f"\tchan1:  {images[i][0]}")
        click.echo(f"\tchan2:  {images[i][1]}")
        click.echo(f"\tlabels: {rois[i]}")
    ans = click.confirm("Are all images/rois correcly matched?", abort=True)

    # train model
    split_axis = None if not split else 0
    api.train_model(model, images, rois, outdir, labels=labelfile, split_axis=split_axis, build_image=True, tempdir=tempdir)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
