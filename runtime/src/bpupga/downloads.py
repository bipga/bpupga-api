"""Download paper inputs and observations into a new user-selected directory."""
from pathlib import Path
import os
import shutil
import tarfile
import tempfile
import urllib.request

ARTIFACT_URL='https://bipga.github.io/supplementary/BiPGA_artifacts.tar.xz'


def download_artifact(destination):
    destination=Path(destination).expanduser().absolute()
    if destination.exists():raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.bipga-download-',dir=destination.parent) as temporary:
        stage=Path(temporary);archive=stage/'artifact.tar.xz';size=0
        with urllib.request.urlopen(ARTIFACT_URL,timeout=60) as response,archive.open('wb') as handle:
            while True:
                block=response.read(1<<20)
                if not block:break
                size+=len(block)
                if size>256*(1<<20):raise ValueError('Artifact exceeds download size limit')
                handle.write(block)
        with tarfile.open(archive,'r:xz') as content:
            members=content.getmembers()
            if len(members)>10000 or sum(m.size for m in members)>2*(1<<30):
                raise ValueError('Artifact exceeds extraction limits')
            for member in members:
                path=Path(member.name)
                if path.is_absolute() or '..' in path.parts or path.parts[0]!='bipga':
                    raise ValueError('Unsafe archive path')
                if not member.isfile() and not member.isdir():raise ValueError('Archive contains a link or special file')
            content.extractall(stage/'extracted',members=members)
        source=stage/'extracted/bipga'
        for name in ['datasets/index.csv','experiments/runs.csv','data/figure_09/PROTOCOL.json']:
            if not (source/name).is_file():raise ValueError('Incomplete paper artifact: '+name)
        source.rename(destination)
    return destination
