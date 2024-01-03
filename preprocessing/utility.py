# Paper Title: A Second Look on BASS - Boosting Abstractive Summarization with Unified Semantic Graphs
# Author: Osman Alperen Koras
# Affiliation: Institute for AI in Medicine (IKIM), University Medicine Essen
# Email: osman.koras@uni-due.de
# Date: 12.10.2023
#
# License: MIT
from __future__ import annotations
import pathlib


def make_directory(directory: pathlib.Path):
    if type(directory) == str:
        directory = pathlib.Path(directory)

    directory.mkdir(parents=True, exist_ok=True)
    return directory.exists()



