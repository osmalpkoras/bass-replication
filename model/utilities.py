# Paper Title: A Second Look on BASS - Boosting Abstractive Summarization with Unified Semantic Graphs
# Author: Osman Alperen Koras
# Affiliation: Institute for AI in Medicine (IKIM), University Medicine Essen
# Email: osman.koras@uni-due.de
# Date: 12.10.2023
#
# License: MIT
from __future__ import annotations

import enum
from datetime import datetime
import math
import os
import pathlib
import torch
from torch.utils.data import DataLoader

from model.models import Bass, RTS2S, exRTS2S
from model.options import Options
from model.generation import beam_search
from model.dataset import BassDataset
from preprocessing.utility import make_directory


def learning_rate_factor(step, warmup):
    if step == 0:
        step = 1
    return min(step ** (-0.5), step * warmup ** (-1.5))


class ModelType(str, enum.Enum):
    NONE = "None"
    Bass = "Bass"
    RTS2S = "RTS2S"
    exRTS2S = "exRTS2S"


def initialize_model(model_type: ModelType, checkpoint_dir: pathlib.Path, options: Options = None):
    if model_type == ModelType.Bass:
        model = Bass(options)
    elif model_type == ModelType.RTS2S:
        model = RTS2S(options)
    elif model_type == ModelType.exRTS2S:
        model = exRTS2S(options)

    make_directory(checkpoint_dir)
    save_model(model, checkpoint_dir / "0.pth")
    return model, 0


def initialize_model_from_weights(model_type: ModelType, weights: pathlib.Path):
    assert model_type != ModelType.NONE
    options = Options()
    if model_type == ModelType.Bass:
        model = Bass(options)
    elif model_type == ModelType.RTS2S:
        model = RTS2S(options)
    elif model_type == ModelType.exRTS2S:
        model = exRTS2S(options)

    weights = torch.load(weights)
    model.load_state_dict(weights)
    return model


def load_model(checkpoint_dir: pathlib.Path, step=None, model=None):
    if step:
        file = checkpoint_dir / f"{step}.pth"
        try:
            loaded_step = int(file.name)
            model = torch.load(file, map_location="cpu")
        except Exception:
            loaded_step = None
            model = initialize_model_from_weights(model, file)
    else:
        file_names = [f.stem for f in checkpoint_dir.glob("*.pth") if f.is_file() and f.stem.isnumeric()]
        file_names.sort(key=lambda x: int(x), reverse=True)
        loaded_step = None
        model = None
        if file_names:
            file_name = file_names[0]
            if step is not None:
                file_name = str(step)
            model = torch.load(checkpoint_dir / f"{file_name}.pth", map_location="cpu")
            loaded_step = int(file_name)

    return model, loaded_step



def save_model(model, checkpoint: pathlib.Path):
    try:
        torch.save(model.module, checkpoint)
    except AttributeError:
        torch.save(model, checkpoint)


def save_model_weights(model, file: pathlib.Path):
    try:
        torch.save(model.module.state_dict(), file)
    except AttributeError:
        torch.save(model.state_dict(), file)


def initialize_training(model_type: ModelType, checkpoint_dir, checkpoint, dataset_dir, options: Options, split, batch_size):
    model, step = load_model(checkpoint_dir, checkpoint, model_type)
    if model is None:
        assert model_type != ModelType.NONE
        model, step = initialize_model(model_type, checkpoint_dir, options)

    print(f"{model_type} checkpoint loaded at step {step}", flush=True)

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
        roberta_optimizer = torch.optim.AdamW([p for n, p in model.module.named_parameters() if n.startswith("text_encoder.")], 
                                              lr=options.optimizer_for_pretrained_part.learning_rate, 
                                              betas=(options.optimizer_for_pretrained_part.beta1, options.optimizer_for_pretrained_part.beta2))

        bass_optimizer = torch.optim.AdamW([p for n, p in model.module.named_parameters() if not n.startswith("text_encoder.")],
                                           lr=options.optimizer_for_other_parts.learning_rate,
                                           betas=(options.optimizer_for_other_parts.beta1, options.optimizer_for_other_parts.beta2),
                                           weight_decay=0.1)
    else:
        roberta_optimizer = torch.optim.AdamW([p for n, p in model.named_parameters() if n.startswith("text_encoder.")], 
                                              lr=options.optimizer_for_pretrained_part.learning_rate,
                                              betas=(options.optimizer_for_pretrained_part.beta1, options.optimizer_for_pretrained_part.beta2))
        bass_optimizer = torch.optim.AdamW([p for n, p in model.named_parameters() if not n.startswith("text_encoder.")], 
                                           lr=options.optimizer_for_other_parts.learning_rate,
                                           betas=(options.optimizer_for_other_parts.beta1, options.optimizer_for_other_parts.beta2),
                                           weight_decay=0.1)
        model.to(options.device)

    training_dataset = BassDataset(dataset_dir, split, device=options.device)
    if split == "train":
        train_loader = DataLoader(training_dataset, batch_size=batch_size, num_workers=os.cpu_count(), shuffle=True, pin_memory=True)
    else:
        train_loader = DataLoader(training_dataset, batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    
    bass_optimizer_scheduler = torch.optim.lr_scheduler.LambdaLR(bass_optimizer, lr_lambda=lambda s: learning_rate_factor(s, options.optimizer_for_other_parts.warmup_steps))
    roberta_optimizer_scheduler = torch.optim.lr_scheduler.LambdaLR(roberta_optimizer, lr_lambda=lambda s: learning_rate_factor(s, options.optimizer_for_pretrained_part.warmup_steps))
    
    for i in range(0, step):
        bass_optimizer_scheduler.step()
        roberta_optimizer_scheduler.step()
    return bass_optimizer, bass_optimizer_scheduler, model, roberta_optimizer, roberta_optimizer_scheduler, train_loader, batch_size, step


def clip_text(summary):
    index = summary.find("</s>")
    if index >= 0:
        summary = summary[:(index + 4)]
    return summary


def parse_time_limit(time_limit: str):
    if time_limit == "00-00:00:00" or time_limit == "00:00:00":
        return math.inf

    days = 0
    if "-" in time_limit:
        time_limit = time_limit.split("-")
        days = int(time_limit[0])
        time_limit = time_limit[1]

    time_limit = datetime.datetime.strptime(time_limit, "%H:%M:%S")
    time_limit = datetime.timedelta(days=days,
                                    hours=time_limit.hour,
                                    minutes=time_limit.minute,
                                    seconds=time_limit.second).total_seconds()
    return time_limit


def calculate_loss(model, batch):
    encoder_input_ids, encoder_padding, decoder_input_ids, decoder_padding, graph_construction_matrix, node_padding, adjacency_matrix = batch
    # Make predictions for this batch
    loss, _, _ = model(encoder_input_ids, encoder_padding, decoder_input_ids, decoder_padding,
                    graph_construction_matrix, node_padding, adjacency_matrix)
    
    return loss


def run_test(model, batch, tokenizer):
    # Make predictions for this batch
    model.eval()
    with torch.no_grad():
        result = beam_search(model, batch, f"cuda:{model.output_device}")
        gs = tokenizer.batch_decode(result[:, :])
        
        generated_summaries = []
        for i in range(0, len(gs)):
            generated_summaries.append(clip_text(gs[i]))
        
    model.train()
    print("test summary: " + generated_summaries[0], flush=True)


class TrainingLimitReached(Exception):
    pass
