"""Command line interface for muscle segmentation."""
from __future__ import annotations
import logging
import pathlib
import re
import sys

import click

from musegai import api


@click.command(context_settings={"show_default": True})
@click.argument("volumes", type=click.Path(exists=True), nargs=-1)
@click.option("-d", "--dest", type=click.Path(), help="Output directory.")
@click.option('-f', '--format', default='.nii.gz', type=click.Choice(['.nii.gz', '.mha', '.mhd', '.hdr']))
@click.option("--model", default="thigh-model3", help="Specify the segmentation model.")
@click.option("--side", default='LR', type=click.Choice(['L', 'R', 'LR', 'NA']), help="Limb's side(s) in image")
@click.option('-v', '--verbose', is_flag=True, help='Show more information')
@click.option("--tempdir", type=click.Path(exists=True), help="Location for temporary files.")
def cli(volumes, dest, format, model, side, tempdir, verbose):
    """Automatic muscle segmentation command line tool.

    \b
    VOLUMES can be:
        - (nothing): show available segmentation models
        - two matching Dixon volumes to segment
        - a single directory with numbered pairs of matching Dixon volumes
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    if not volumes:
        # no argument: list available models
        click.echo("Available segmentation models:")
        for available_model in api.list_models():
            click.echo(f"\t{available_model}")
        sys.exit(0)

    if (len(volumes) == 1) and pathlib.Path(volumes[0]).is_dir():
        # a folder with volume pairs
        root = pathlib.Path(volumes[0])
        regex = re.compile(r"(.+?)(\d+).[\w.]+$")
        volumes = {}
        for file in sorted(root.glob("*")):
            match = regex.match(file.name)
            if not match:
                continue
            name, _ = match.groups()
            volumes.setdefault(name, []).append(file)
            if len(volumes[name]) > 2:
                click.echo(f"Expecting two volume files with prefix: {name}")
        click.echo(f"Found {len(volumes)} volume pair(s) to segment:")
        for name in volumes:
            click.echo(f"\t{name}")

    elif len(volumes) == 2 and all(pathlib.Path(file).is_file() for file in volumes):
        # individual files
        root = "."
        volumes = [pathlib.Path(file) for file in volumes]
        name = volumes[0].name
        volumes = {name: volumes}
        click.echo(f"Found one volume pair to segment: {name}")

    else:
        # invalid number of arguments
        click.echo("Expecting two volume files or a single directory")
        sys.exit(0)

    # destination
    dest = pathlib.Path(root if dest is None else dest)
    dest.mkdir(exist_ok=True, parents=True)
    destfiles = {name: (dest / name).with_suffix(format) for name in volumes}
    for name in list(destfiles):
        if destfiles[name].is_file():
            click.echo(f"Output file already exists: {destfiles[name]}, skipping")
            volumes.pop(name)
            destfiles.pop(name)

    if not volumes:
        click.echo("Nothing to do.")
        sys.exit(0)

    # side
    if side == 'NA':
        side = None

    # segment volumes
    click.echo(f"Segmenting {len(volumes)} volume(s)...")
    inputs = list(volumes.values())
    outputs = list(destfiles.values())
    api.run_model(model, inputs, outputs, side=side, tempdir=tempdir)

    click.echo("Done.")


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
