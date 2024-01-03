# Paper Title: A Second Look on BASS - Boosting Abstractive Summarization with Unified Semantic Graphs
# Author: Osman Alperen Koras
# Affiliation: Institute for AI in Medicine (IKIM), University Medicine Essen
# Email: osman.koras@uni-due.de
# Date: 12.10.2023
#
# License: MIT
from __future__ import annotations

import concurrent
import enum
import gzip
import operator
import os
import pathlib
import pickle
import signal
import time
from functools import reduce

from abc import abstractmethod
from concurrent.futures import wait, FIRST_COMPLETED, Future

import rich.traceback
import stanza.protobuf
import stanza.server
import torch
from stanza.protobuf import CoreNLP_pb2
from transformers import BatchEncoding, RobertaTokenizerFast, AutoTokenizer

from preprocessing import javaobj
from preprocessing.datasets import Dataset, Datapoint, CoreNlpDatapoint, BassDatapoint
from preprocessing.doc import Document
from preprocessing.parser import CoreNlpAnnotationParser
from preprocessing.utility import make_directory


class PreprocessorType(str, enum.Enum):
    USGsrc = "USGsrc"
    USGppr = "USGppr"


class ParallelizationType(str, enum.Enum):
    ByFile = "ByFile"
    ByDatapoint = "ByDatapoint"


class DatasetPreprocessor:
    def __init__(self,
                 pool_index: int = 0,
                 input_directory: str = "data/bigPatentData",
                 output_directory: str = "data/preprocessed/bigPatentData",
                 timeout: int = 60,
                 break_file: pathlib.Path = None):
        self.input_directory: pathlib.Path = pathlib.Path(input_directory)
        self.output_directory: pathlib.Path = pathlib.Path(output_directory)
        self.pool_index = pool_index
        self.break_file: pathlib.Path = break_file
        self.timeout = timeout
        self.datapoint_loader: DatapointLoader = None

    def should_stop_preprocessing(self) -> bool:
        if self.break_file and self.break_file.exists():
            return True

        return False

    def process_datapoint(self, input_datapoint: Datapoint) -> tuple[Datapoint, float]:
        print(f">>> worker {self.pool_index} now parsing {input_datapoint.id}", flush=True)

        try:
            start_time = time.time()
            output_datapoint = self.parse_datapoint(input_datapoint)
            end_time = time.time()
            return output_datapoint, end_time - start_time
        except Exception as ex:
            rich.traceback.Console().print_exception()
            print(f">>> worker {self.pool_index} failed to parse {input_datapoint.id}", flush=True)
            raise

    def run(self, batches: list[list[pathlib.Path]], worker_count: int = os.cpu_count(), parallelization_type: ParallelizationType = ParallelizationType.ByDatapoint):
        print(f"main process id: {os.getpid()}")
        preprocessor_locals = self.before_run(batches)
        if worker_count > 1:
            if parallelization_type == ParallelizationType.ByDatapoint:
                self._run_parallelized_internal(DatasetPreprocessor._run_datapoint_internal, reduce(operator.add, batches), worker_count)
            elif parallelization_type == ParallelizationType.ByFile:
                self._run_parallelized_internal(DatasetPreprocessor._run_internal, (item for item in batches), worker_count)
        else:
            for batch in batches:
                if self.break_file and self.break_file.exists():
                    print("found break file... now stopping preprocessing.")
                    break
                self._run_internal(batch)

    def _run_parallelized_internal(self, method, generator, worker_count: int):
        total_datapoints = 0
        total_time = 0.0
        futures: set[Future] = set()
        done: set[Future]
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            for item in generator:
                futures.add(executor.submit(method, self, item))

            while futures:
                if self.break_file and self.break_file.exists():
                    print("found break file... now stopping preprocessing.")
                    executor.shutdown(wait=False, cancel_futures=False)

                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        # else get the timings
                        t, n = future.result()
                        total_datapoints += n
                        total_time += t
                    except Exception:
                        # if a worker terminated with an exception, reraise the exception and stop preprocessing
                        print(f">>> a worker encountered a critical error. terminating preprocessing...")
                        raise

            executor.shutdown(wait=False, cancel_futures=True)

    class TimeoutException(BaseException):
        pass

    @staticmethod
    def raise_preprocessing_timeout_exception(signum, frame):
        raise DatasetPreprocessor.TimeoutException()

    def _run_internal(self, files: list[pathlib.Path]) -> tuple[float, int]:
        total_datapoints = 0
        total_time = 0.0
        
        for file in files:        
            try:
                if self.break_file and self.break_file.exists():
                    print("found break file... now stopping preprocessing.")
                    break
                tdp, tt = self._run_datapoint_internal(file)
                total_datapoints += tdp
                total_time += tt
            except Exception:
                pass

        if total_datapoints == 0:
            print(f">>> worker {self.pool_index} did nothing...")
        else:
            print(f">>> worker {self.pool_index} took {total_time} seconds in total for all datapoints ({total_datapoints}) and "
                  f"{total_time/total_datapoints} seconds at average for each datapoint.")

        return total_time, total_datapoints

    def _run_datapoint_internal(self, file) -> tuple[float, int]:
        total_datapoints = 0
        total_time = 0

        if self.break_file and self.break_file.exists():
            print("found break file... now stopping preprocessing.")
        else:
            self.on_run(file)
            signal.signal(signal.SIGALRM, DatasetPreprocessor.raise_preprocessing_timeout_exception)
            
            with self.datapoint_loader.open_datafile(file) as fin:
                print(f">>> now parsing [ {file} ]", flush=True)
                if worker_data := self.datapoint_loader.load_datapoint(fin):
                    print(f">>> now preprocessing {worker_data.id}", flush=True)

                    try:
                        start_time = time.time()

                        signal.alarm(self.timeout)
                        try:
                            datapoint = self.parse_datapoint(worker_data)
                        except DatasetPreprocessor.TimeoutException:
                            print(f">>> preprocessing timed out for {worker_data.id}...", flush=True)
                            datapoint = None
                        finally:
                            signal.alarm(0)
                        end_time = time.time()
                        total_time += (end_time - start_time)
                        total_datapoints += 1
                    except Exception:
                        print(f">>> a worker encountered a critical error. terminating preprocessing...")
                        raise

                    if datapoint:
                        self.save_datapoint(file, datapoint)
                    else:
                        print(f">>> failed to parse {worker_data.id}", flush=True)
                    
        return total_datapoints, total_time


    def save_datapoint(self, file: pathlib.Path, datapoint: Datapoint):
        try:
            with gzip.open(self.output_directory / file, "wb") as file_out:
                pickle.dump(datapoint, file_out)
            
            return
        except Exception:
            rich.traceback.Console().print_exception()

    def append_to_dataset(self, file, name, index, tensors):
        numpy_array = tensors.unsqueeze(0).numpy()
        if name not in file:
            dataset_size = (file.attrs["max_length"],) + numpy_array.shape[1:]
            dset = file.create_dataset(name, dataset_size, numpy_array.dtype)
        else:
            dset = file[name]

        dset[index, ...] = numpy_array

    @abstractmethod
    def before_run(self, dataset: Dataset) -> any:
        return None

    @abstractmethod
    def on_run(self, file):
        pass


    @abstractmethod
    def parse_datapoint(self, datapoint: Datapoint) -> Datapoint:
        """
            parses given datapoint and returns another datapoint, ready to be parsed by another preprocessor or ...
            returns None if errors occured during the parsing. raises an exception iff the preprocessor should stop
            execution at once.
        :param datapoint:
        """
        pass


class DatapointLoader:    
    def __init__(self, input_directory: str, output_directory: str):
        self.input_directory = input_directory
        self.output_directory = output_directory

    @abstractmethod
    def open_datafile(self, file: pathlib.Path):
        pass

    @abstractmethod
    def load_datapoint(self, fin) -> Datapoint:
        """
            loads and returns the datapoint from fin, None if errors happen or there is no datapoint to load
        """
        pass


class JavaDatapointLoader(DatapointLoader):
    def __init__(self, input_directory: str, output_directory: str):
        super().__init__(input_directory=input_directory, output_directory=output_directory)
        self.reader = None
        
    def open_datafile(self, file: pathlib.Path):
        path = self.output_directory / file
        make_directory(path.parents[0])

        self.reader = None
        return gzip.open(self.input_directory / file, 'rb')

    def load_datapoint(self, fin) -> CoreNlpDatapoint:
        if not self.reader:
            self.reader = javaobj.JavaObjectStreamReader()
            
        while True:
            try:
                obj = self.reader.load(fin)
                return obj
            except EOFError:
                break
            except Exception as ex:
                rich.traceback.Console().print_exception()

        return None


# parse a java datapoint (converted to python) where the document unified semantic graph has been created
# using the original code of the authors
class USGsrcPreprocessor(DatasetPreprocessor):
    def __init__(self,
                 input_directory: str,
                 output_directory: str,
                 timeout: int,
                 token_max_length: int = 1024,
                 node_max_length: int = 1024,
                 break_file: pathlib.Path = None):
        super().__init__(input_directory=input_directory,
                         output_directory=output_directory,
                         break_file=break_file,
                         timeout=timeout)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.parser: CoreNlpAnnotationParser = None
        self.tokenizer: RobertaTokenizerFast = None
        self.datapoint_loader = JavaDatapointLoader(input_directory=input_directory, output_directory=output_directory)

        self.token_max_length: int = token_max_length
        self.node_max_length: int = node_max_length

    def before_run(self, dataset: Dataset) -> any:
        pass

    def on_run(self, file):
        if self.parser is None:
            self.parser = CoreNlpAnnotationParser()
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base", model_max_length=10000)


    def parse_datapoint(self, obj: CoreNlpDatapoint) -> BassDatapoint:
        try:
            datapoint = CoreNlpDatapoint.parse_from(obj, include_protobuf_annotation=False)
            tokenized_input: BatchEncoding = self.tokenizer(datapoint.document_text, return_tensors="pt")
            tokenized_output: BatchEncoding = self.tokenizer(datapoint.summary_text, return_tensors="pt")

            # there are some duplicate tokens in a sentence - how did this happen? how can I fix it?
            document: Document = Document.build_from_corenlpdatapoint(datapoint)
            gcm, gam = document.export_graph_information_for_tokenization(tokenized_input)

            dp: BassDatapoint = BassDatapoint()
            dp.input_tokens = tokenized_input.input_ids
            dp.output_tokens = tokenized_output.input_ids
            dp.graph_construction_matrix = gcm
            dp.graph_adjacency_matrix = gam
            dp.id = datapoint.id
            
            return self.clip_length(dp)
        except Exception:
            rich.traceback.Console().print_exception()
            return None

    def clip_length(self, datapoint: CoreNlpDatapoint):
        node_padding = torch.ones(self.token_max_length, dtype=torch.long)
        gcm = datapoint.graph_construction_matrix.float()
        gcm = gcm[:, :self.token_max_length]
        dims = gcm.size()
        if dims[1] < self.token_max_length:
            gcm = torch.cat([gcm, torch.zeros((dims[0], self.token_max_length - dims[1]))], 1)

        meaningful_nodes = torch.sum(gcm, 1).bool()
        meaningful_nodes[0] = True # first row represents the supernode
        gcm = gcm[meaningful_nodes, :]
        gcm = gcm[:self.node_max_length, :]
        dims = gcm.size()
        if dims[0] < self.node_max_length:
            gcm = torch.cat([gcm, torch.zeros((self.node_max_length - dims[0], dims[1]))], 0)
            node_padding[dims[0]:] = 0

        gam = datapoint.graph_adjacency_matrix[meaningful_nodes, :][:, meaningful_nodes]
        gam = gam[:self.node_max_length, :self.node_max_length]
        if gam.size(0) < self.node_max_length:
            gam = torch.cat(
                [gam, torch.zeros((self.node_max_length - gam.size(0), gam.size(1)), dtype=torch.long)], 0)
            gam = torch.cat([gam, torch.zeros((self.node_max_length, self.node_max_length - gam.size(1)),
                                              dtype=torch.long)], 1)

        token_in = datapoint.input_tokens.squeeze()[:self.token_max_length]
        token_in_padding = torch.ones(self.token_max_length, dtype=torch.long)
        if token_in.size(0) < self.token_max_length:
            token_in_padding[token_in.size(0):] = 0
            token_in = torch.cat(
                [token_in, torch.ones(self.token_max_length - token_in.size(0), dtype=torch.long)], 0)

        token_out = datapoint.output_tokens.squeeze()[:self.token_max_length]
        token_out_padding = torch.ones(self.token_max_length, dtype=torch.long)
        if token_out.size(0) < self.token_max_length:
            token_out_padding[token_out.size(0):] = 0
            token_out = torch.cat(
                [token_out, torch.ones(self.token_max_length - token_out.size(0), dtype=torch.long)], 0)

        datapoint.input_tokens = token_in.long()
        datapoint.input_padding = token_in_padding.float()
        datapoint.output_tokens = token_out.long()
        datapoint.output_padding = token_out_padding.float()
        datapoint.graph_construction_matrix = gcm.float()
        datapoint.node_padding = node_padding.float()
        datapoint.graph_adjacency_matrix = gam.int()
        return datapoint


class USGpprPreprocessor(USGsrcPreprocessor):
    def parse_datapoint(self, obj: CoreNlpDatapoint) -> BassDatapoint:
        try:
            datapoint = CoreNlpDatapoint.parse_from(obj, include_protobuf_annotation=True, include_documents=False)
            tokenized_input: BatchEncoding = self.tokenizer(datapoint.document_text, return_tensors="pt")
            tokenized_output: BatchEncoding = self.tokenizer(datapoint.summary_text, return_tensors="pt")

            for annotation in datapoint.protobuf_annotation:
                doc = CoreNLP_pb2.Document()
                stanza.protobuf.parseFromDelimitedString(doc, annotation)
                document: Document = self.parser.parse(doc)
                document.text = datapoint.document_text
                datapoint.annotation.append(document)

            document = Document(datapoint.document_text)
            for annotation in datapoint.annotation:
                document.merge(annotation)
            document.text = datapoint.document_text
            document.build_strictly_compliant_with_bass_paper()
            gcm, gam = document.export_graph_information_for_tokenization(tokenized_input)

            dp: BassDatapoint = BassDatapoint()
            dp.input_tokens = tokenized_input.input_ids
            dp.output_tokens = tokenized_output.input_ids
            dp.graph_construction_matrix = gcm
            dp.graph_adjacency_matrix = gam
            dp.id = datapoint.id

            return self.clip_length(dp)
        except Exception:
            rich.traceback.Console().print_exception()
            return None