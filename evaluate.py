# Paper Title: A Second Look on BASS - Boosting Abstractive Summarization with Unified Semantic Graphs
# Author: Osman Alperen Koras
# Affiliation: Institute for AI in Medicine (IKIM), University Medicine Essen
# Email: osman.koras@uni-due.de
# Date: 12.10.2023
#
# License: MIT
from __future__ import annotations

from datetime import datetime
import gzip
import pathlib
import pickle

import rich.traceback
import click
from preprocessing.utility import make_directory

from model.evaluation import Evaluation


@click.command()
@click.argument("model-output-file",
                type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, writable=False,
                                path_type=pathlib.Path, resolve_path=True),
                help="the model output file generated by generate.py")
@click.argument("evaluation-output-dir",
                type=click.Path(exists=False, file_okay=False, dir_okay=True, readable=True, writable=True,
                                path_type=pathlib.Path, resolve_path=True),
                help="the directory in which the scores should be saved. the file naming convention is be $output-file.$score")
@click.option("--output-file", default=None, type=str, help="the prefix for the files generated by this script. if not given, this will be set to the model-output file name.")
def main(model_output_file: pathlib.Path, evaluation_output_dir: pathlib.Path, output_file: str):
    print(datetime.now())
    try:
        make_directory(evaluation_output_dir)
        if output_file is None:
            output_file = model_output_file.name

        evaluation = Evaluation()
        evaluation.load(model_output_file)
        print(evaluation.run())

        scores = {"r1": evaluation.get_r1_scores(),
                  "r2": evaluation.get_r2_scores(),
                  "rL": evaluation.get_rL_scores(),
                  "rLsum": evaluation.get_rLsum_scores(),
                  "bs": evaluation.get_bert_scores()}

        for score, values in scores.items():
            with gzip.open(evaluation_output_dir / (output_file + f".{score}"), "w+") as fin:
                pickle.dump(values, fin)

    except Exception:
        rich.traceback.Console().print_exception()


if __name__ == '__main__':
    main()



