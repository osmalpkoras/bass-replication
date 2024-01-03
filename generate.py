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
import os
import pathlib
import time
import pickle

import rich.traceback
import click
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from model.generation import beam_search
from model.dataset import BassDataset
from model.utilities import clip_text, load_model, ModelType
from preprocessing.utility import make_directory


@click.command()
@click.argument("dataset_dir",
                type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, writable=True,
                                path_type=pathlib.Path, resolve_path=True),
                help="the directory of the dataset that contains the pre-processed test split.")
@click.argument("checkpoint-dir",
                type=click.Path(exists=False, file_okay=False, dir_okay=True, readable=True, writable=True,
                                path_type=pathlib.Path, resolve_path=True),
                help="the directory of model checkpoints")
@click.option("--output-file", default="output", type=str, help="the file name ending of the generation output.")
@click.option("--batch-size", default=8, type=int, help="the batch size for beam search (with beam width = 5)")
@click.option("--device", default=None, type=str, help="the device to generate summaries on.")
@click.option("--checkpoint", default=None, type=str, help="the exact checkpoint name to load. if not given, the script will try to load the highest numerical file name which is the model with the latest checkpoint (by our convention, numeric file names correspond to training steps).")
@click.option("--model", default=ModelType.NONE, type=click.Choice(ModelType, case_sensitive=False), help="the type of model to load. this is required for loading model weights instead of model checkpoints.")
def main(dataset_dir: pathlib.Path, checkpoint_dir: pathlib.Path, output_file: pathlib.Path, batch_size: int,
         device: str, checkpoint: int, model: ModelType):
    print(datetime.now())
    try:
        make_directory(checkpoint_dir)
        split = "test"

        model, step = load_model(checkpoint_dir, checkpoint, model)
        options = model.options
        options.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained("roberta-base", model_max_length=options.transformer_options.max_input_length)
        if options.device.startswith("cuda:"):
            os.environ["CUDA_VISIBLE_DEVICES"] = options.device.split(":")[1]

        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join([str(i) for i in range(0, torch.cuda.device_count())]))
        if options.device.startswith("cuda"):
            device_ids = []
            for index, i in enumerate([i for i in visible_devices.split(",")]):
                if i in options.device:
                    device_ids.append(index)
                    continue

                ret = torch.cuda.list_gpu_processes(device=int(i))
                if "no processes are running" in ret:
                    device_ids.append(index)
            options.device = "cuda"

            print(f"using the following cuda devices: {device_ids}", flush=True)
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            model.to(f"{options.device}:{device_ids[0]}")
        else:
            model.to(options.device)

        training_dataset = BassDataset(dataset_dir, split, device=device)
        train_loader = DataLoader(training_dataset, batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
        model.eval()

        start_time = time.time()
        for batch_index, batch in enumerate(train_loader):
            with torch.no_grad():
                failed = False
                start_bi_time = time.time()
                try:
                    result = beam_search(model, batch, options)

                    finished_summaries = torch.any(result == 2, dim=-1).to("cpu")
                    d = tokenizer.batch_decode(batch[0][finished_summaries, :])
                    s = tokenizer.batch_decode(batch[2][finished_summaries, :])
                    gs = tokenizer.batch_decode(result[finished_summaries, :])

                    documents = []
                    summaries = []
                    generated_summaries = []
                    for i in range(0, len(d)):
                        documents.append(clip_text(d[i]))
                        summaries.append(clip_text(s[i]))
                        generated_summaries.append(clip_text(gs[i]))
                except Exception:
                    rich.traceback.Console().print_exception()
                    failed = True

                end_bi_time = time.time()
                if failed:
                    print(f"batch {batch_index} failed ({end_bi_time - start_bi_time} seconds)", flush=True)
                else:
                    if torch.all(finished_summaries):
                        print(f"batch {batch_index} completed ({end_bi_time - start_bi_time} seconds)",
                              flush=True)
                    else:
                        print(
                            f"batch {batch_index} completed ({end_bi_time - start_bi_time} seconds) with {int(torch.sum(finished_summaries == False))} unfinished summaries",
                            flush=True)

                    with gzip.open(checkpoint_dir / f"{checkpoint}.{output_file}", "ab+") as str_out:
                        pickle.dump({"documents": documents, "summaries": summaries,
                                     "generated_summaries": generated_summaries}, str_out)

        end_time = time.time()
        print(f"finished generation after {end_time - start_time} seconds...", flush=True)

    except Exception:
        rich.traceback.Console().print_exception()
    finally:
        pass


if __name__ == '__main__':
    main()



