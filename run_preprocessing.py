# Paper Title: A Second Look on BASS - Boosting Abstractive Summarization with Unified Semantic Graphs
# Author: Osman Alperen Koras
# Affiliation: Institute for AI in Medicine (IKIM), University Medicine Essen
# Email: osman.koras@uni-due.de
# Date: 12.10.2023
#
# License: MIT
from __future__ import annotations
import multiprocessing

import gzip
import pathlib
import pickle
import shutil
import time
import uuid

from datetime import datetime

import rich.traceback
import click

from preprocessing.datasets import Dataset, Datapoint
from preprocessing.preprocessors import DatasetPreprocessor, ParallelizationType, \
    PreprocessorType, USGsrcPreprocessor, USGpprPreprocessor
from preprocessing.utility import make_directory


@click.command()
@click.argument("source_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, writable=True,
                                              path_type=pathlib.Path, resolve_path=True),
                help="the root directory of the source dataset.")
@click.argument("temp_dir", type=click.Path(exists=False, file_okay=False, dir_okay=True, readable=True, writable=True,
                                            path_type=pathlib.Path, resolve_path=True),
                help="a temporary directory used for intermediate files. useful for compute clusters with networked hard drives.")
@click.argument("target_dir", type=click.Path(exists=False, file_okay=False, dir_okay=True, readable=True, writable=True,
                                              path_type=pathlib.Path, resolve_path=True),
                help="the root directory into which the preprocessed dataset should be output.")
@click.argument("preprocessor-type", type=click.Choice(PreprocessorType, case_sensitive=False),
                help="the kind of preprocessing to perform.")
@click.option("--pool-index", "-i", default=0, help="The index of the pool worker pre-processing the dataset.")
@click.option("--pool-size", "-I", default=1, help="The total number of pool workers pre-processing the dataset.")
@click.option("--worker-count", "-I", default=1, help="The number of workers to use for parallelization.")
@click.option("--timeout", "-t", default=120*60*1000, help="The timeout for preprocessing a single datapoint")
@click.option("--parallelization-type", default=ParallelizationType.ByDatapoint, type=click.Choice(ParallelizationType, case_sensitive=False),
              help="The type of parallelization, if more than one worker is used.")
def main(source_dir: pathlib.Path, temp_dir: pathlib.Path, target_dir: pathlib.Path, preprocessor_type: click.Choice, pool_index, pool_size, worker_count, timeout, parallelization_type):
    try:
        print("start process...")
        print(f"source_dir {source_dir}")
        print(f"temp_dir {temp_dir}")
        print(f"target_dir {target_dir}")
        print(f"--pool-index {pool_index}")
        print(f"--pool-size {pool_size}")
        print(f"--worker-count {worker_count}")
        print(f"--timeout {timeout}")
        print(f"--parallelization-type {parallelization_type}")
        print("==============================", flush=True)

        files = get_batch_of_missing_datapoints(source_dir, target_dir, pool_index, pool_size)

        print(f"start preprocessing {len(files)} files...", flush=True)

        subdir = temp_dir / str(uuid.uuid1())
        while subdir.exists():
            subdir = temp_dir / str(uuid.uuid1())
        temp_dir = subdir

        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"

        if not prepare_temp_dir(files, source_dir, temp_dir, target_dir, input_dir, output_dir):
            return

        preprocessor = initialize_preprocessor(input_dir, output_dir, preprocessor_type, target_dir, timeout)
        start_time = time.time()
        print(f"preprocessing is starting at {datetime.fromtimestamp(start_time).strftime('%H:%M:%S, %d.%m.%Y')}", flush=True)
        failed = False
        try:
            preprocessor.run([files], worker_count, parallelization_type)
        except Exception:
            rich.traceback.Console().print_exception()
            failed = True
        end_time = time.time()

        # merge_datapoints(dataset, intermediate_dir, output_dir)

        if not copy_datafiles(files, output_dir, target_dir, copy_indices=True):
            print("quit process, because files could not be copied from output_dir to target_dir")
            return

        if failed:
            print(f"preprocessing failed after {end_time-start_time} seconds", flush=True)
        else:
            print(f"preprocessing finished successfully in {end_time-start_time} seconds", flush=True)
    except Exception:
        rich.traceback.Console().print_exception()
        pass
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def initialize_preprocessor(input_dir, output_dir, preprocessor_type, target_dir, timeout):
    preprocessor: DatasetPreprocessor = None
    if preprocessor_type == PreprocessorType.USGppr:
        preprocessor = USGpprPreprocessor(input_directory=str(input_dir),
                                          output_directory=str(output_dir),
                                          timeout=timeout,
                                          break_file=target_dir / "stop_preprocessing")
    elif preprocessor_type == PreprocessorType.USGsrc:
        preprocessor = USGsrcPreprocessor(input_directory=str(input_dir),
                                          output_directory=str(output_dir),
                                          timeout=timeout,
                                          break_file=target_dir / "stop_preprocessing")
    return preprocessor


def get_batch_of_missing_datapoints(source_dir: pathlib.Path, target_dir: pathlib.Path, pool_index: int, pool_size: int):    
    missing_datapoints = []
    
    index_file = target_dir / ".index"
    if index_file.exists():
        print(f"found an index at {str(index_file)}", flush=True)
        with gzip.open(index_file, "rb+") as fin:
            datapoints = pickle.load(fin)
            
        print(f"loaded {len(datapoints)} datapoints...")
        
        for file in datapoints:
            if not (target_dir / file).exists():
                missing_datapoints.append(file)
        
        print(f"found {len(missing_datapoints)} missing datapoints...")
    else:
        print(f"file {str(index_file)} not found...", flush=True)
        files = [f.relative_to(source_dir) for f in source_dir.rglob("*.gz")] 
        files.sort(key=lambda x: str(x))
        
        for file in files:
            batch = []
            source_file = source_dir / file
            target_file = target_dir / file
            source_files = [f.relative_to(source_dir) for f in source_file.iterdir() if f.is_file() and f.name.isnumeric()]
            if not target_file.is_dir():
                batch.extend(source_files)
            else:
                for dp in source_files:
                    if not (target_file / dp.name).exists():
                        batch.append(dp)

            batch.sort(key=lambda x: int(x.name))
            missing_datapoints.extend(batch)
        
        make_directory(index_file.parent)
        with gzip.open(index_file, "wb+") as fout:
            pickle.dump(missing_datapoints, fout)
        
        print(f"created an index file at {str(index_file)}", flush=True)

    return missing_datapoints[pool_index::pool_size]


def merge_datapoints(dataset, source_dir, target_dir):
    # get all datapoints in output_dir
    existing_datapoints = extract_existing_datapoints(dataset, target_dir)
    # merge from intermediate_dir to output_dir
    for file in dataset.files:
        filepath = source_dir / file.path
        if filepath.exists():
            with gzip.open(str(filepath.resolve()), "rb+") as file_in:
                try:
                    while True:
                        datapoint: Datapoint = pickle.load(file_in)
                        if datapoint.id in existing_datapoints:
                            print(f"target_dir already contains {datapoint.id}")
                            continue
                        else:
                            existing_datapoints.append(datapoint.id)

                        destination = target_dir / file.path
                        make_directory(destination.parents[0])
                        with gzip.open(destination, "ab+") as file_out:
                            pickle.dump(datapoint, file_out)
                except EOFError:
                    pass
                except Exception:
                    rich.traceback.Console().print_exception()

    index_file = target_dir / (str(uuid.uuid1()) + ".index")
    while index_file.exists():
        index_file = target_dir / (str(uuid.uuid1()) + ".index")

    index = Dataset()
    for dp in existing_datapoints:
        index.add(dp.split(":")[0], dp.split(":")[1])

    index.save(str(index_file.resolve()))


def prepare_temp_dir(files: list[pathlib.Path],
                     source_dir: pathlib.Path,
                     temp_dir: pathlib.Path,
                     target_dir: pathlib.Path,
                     input_dir: pathlib.Path,
                     output_dir: pathlib.Path):
    if not make_directory(temp_dir):
        print("quit process, because temp directory does neither exist nor could be created.")
        return False

    if any(temp_dir.iterdir()):
        print("quit process, because temp directory is not empty")
        return False

    if not make_directory(target_dir):
        print("quit process, because target directory does neither exist nor could be created.")
        return False

    make_directory(input_dir)
    make_directory(output_dir)

    if not copy_datafiles(files, source_dir, input_dir):
        print("quit process, because files could not be copied from source_dir")
        return False

    # if not copy_datafiles(files, target_dir, output_dir):
    #     print("quit process, because files could not be copied from target_dir")
    #     return False

    return True


def extract_existing_datapoints(dataset, output_dir):
    existing_datapoints = []
    for file in dataset.files:
        filepath = output_dir / file.path
        if filepath.exists():
            with gzip.open(str(filepath.resolve()), "rb+") as file_in:
                try:
                    while True:
                        datapoint: Datapoint = pickle.load(file_in)
                        if datapoint.id not in existing_datapoints:
                            existing_datapoints.append(datapoint.id)
                except EOFError:
                    pass
                except Exception:
                    rich.traceback.Console().print_exception()
    return existing_datapoints


def copy_datafiles(files, source_dir, target_dir, copy_indices=False):
    try:
        
        pool = multiprocessing.Pool()
        pool.starmap(copy_datafile, [(source_dir, target_dir, file) for file in files])
        pool.close()
        pool.join()

        if copy_indices:
            indices = pathlib.Path(source_dir).rglob("*.index")
            for file in indices:
                shutil.copyfile(file, target_dir / file.name)

        return True
    except Exception:
        return False


def copy_datafile(source_dir: pathlib.Path, target_dir: pathlib.Path, file: pathlib.Path):
    try:
        source = source_dir / file
        destination = target_dir / file
        if source.is_file():
            make_directory(destination.parents[0])
            # print(f"now copying {file}", flush=True)
            shutil.copyfile(source, destination)
        elif source.is_dir():
            make_directory(destination)
            print(f"now copying from {source} to {destination}...", flush=True)
            for f in (x for x in source.iterdir() if x.is_file):
                # print(f"now copying {f}", flush=True)
                shutil.copyfile(source / f, destination / f)
    except Exception:
        rich.traceback.Console().print_exception()
    


if __name__ == '__main__':
    main()



