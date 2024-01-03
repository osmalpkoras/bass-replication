# Paper Title: A Second Look on BASS - Boosting Abstractive Summarization with Unified Semantic Graphs
# Author: Osman Alperen Koras
# Affiliation: Institute for AI in Medicine (IKIM), University Medicine Essen
# Email: osman.koras@uni-due.de
# Date: 12.10.2023
#
# License: MIT
from __future__ import annotations

from datetime import datetime, timedelta
import itertools
import math
import pathlib
import time
import rich.traceback
import click
import torch
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer

from model.options import Options
from model.utilities import calculate_loss, clip_text, initialize_training, parse_time_limit, run_test, save_model, \
    ModelType, TrainingLimitReached
from preprocessing.utility import make_directory


@click.command()
@click.argument("dataset_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, writable=True,
                                               path_type=pathlib.Path, resolve_path=True),
                help="the directory containing the dataset with the train split.")
@click.argument("checkpoint-dir", type=click.Path(exists=False, file_okay=False, dir_okay=True, readable=True, writable=True,
                                                  path_type=pathlib.Path, resolve_path=True),
                help="the directory to save model checkpoints to.")
@click.option("--time-limit", default="00-00:00:00", help="the time limit after which the training should terminate gracefully.")
@click.option("--log-dir", default="log/0", type=str, help="the directory to save log files to.")
@click.option("--step-limit", default=0, type=int, help="the step limit after which the training should terminate gracefully.")
@click.option("--checkpoint-step-size", default=10000, type=int, help="the number of steps to perform before saving another model checkpoint.")
@click.option("--batch-size", default=8, type=int, help="the batch size for a single training step.")
@click.option("--accumulate-gradients", default=1, type=int, help="the number of batches to accumulate for each step. the effective batch size per step will be accumulate_gradients * batch_size.")
@click.option("--max-epochs", default=0, type=int, help="the maximum number of epochs to iterate through the training set.")
@click.option("--device", default=None, type=str, help="the device to train on.")
@click.option("--checkpoint", default=-1, type=int, help="the specific checkpoint to continue training with. this can also refer to model weights.")
@click.option("--model", default=ModelType.NONE, type=click.Choice(ModelType, case_sensitive=False), help="the model type to initialize. this is required if no prior checkpoint is loaded or when model weights are loaded.")
def main(dataset_dir: pathlib.Path, time_limit: str, step_limit: int, checkpoint_dir: pathlib.Path, checkpoint_step_size: int, batch_size: int, accumulate_gradients: int, max_epochs: int, device: str, log_dir: pathlib.Path, checkpoint: int, model: ModelType):
    print(f"Script starting time: {datetime.now()}")
    try:
        make_directory(checkpoint_dir)
        if step_limit <= 0:
            step_limit = math.inf

        split = "train"
        epochs = itertools.count() if max_epochs <= 0 else range(0, max_epochs)
        time_limit = parse_time_limit(time_limit)

        options = Options()
        options.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device detected: {options.device}", flush=True)
        
        bass_optimizer, bass_optimizer_scheduler, model, roberta_optimizer, roberta_optimizer_scheduler, train_loader, batch_size, step = initialize_training(model, checkpoint_dir, checkpoint, dataset_dir, options, split, batch_size)
        tokenizer = AutoTokenizer.from_pretrained("roberta-base", model_max_length=options.transformer_options.max_input_length)

        model.train()
        roberta_optimizer.zero_grad()
        bass_optimizer.zero_grad()
        writer = SummaryWriter(log_dir=str(checkpoint_dir/ log_dir))

        running_loss = 0
        batch_accumulation = 0
        start_time = time.time()
        last_step_time = start_time
        test_instance = None  # for tracking the learning process
        
        try:    
            for epoch in epochs:
                print(f"starting epoch {epoch}:")
                start_time_epoch = time.time()
                
                for batch_index, batch in enumerate(train_loader):
                    if not test_instance:
                        test_instance = tuple(i[0:1,:] for i in batch)
                        print("test document: " + clip_text(tokenizer.decode(test_instance[0].squeeze())), flush=True)
                        print("test summary: " + clip_text(tokenizer.decode(test_instance[2].squeeze())), flush=True)

                    loss = calculate_loss(model, batch)
                    loss.backward(torch.ones(loss.size(), device=loss.device))
                    # Gather data and report
                    scalar_loss = loss.mean().item()
                    writer.add_scalar("training_loss", scalar_loss, step)
                    running_loss += scalar_loss * batch_size
                    batch_accumulation += 1

                    if batch_accumulation == accumulate_gradients:
                        batch_accumulation = 0

                        # Adjust learning weights
                        if options.maximum_gradient_norm >= 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), options.maximum_gradient_norm)

                        bass_optimizer.step()
                        bass_optimizer_scheduler.step()

                        roberta_optimizer.step()
                        roberta_optimizer_scheduler.step()

                        roberta_optimizer.zero_grad()
                        bass_optimizer.zero_grad()

                        if step % 1000 == 0:
                            run_test(model, test_instance, tokenizer)

                        step += 1
                        if step % 100 == 0:
                            last_loss = running_loss / 100 / batch_size  # loss per batch
                            print(f"batch {step} loss: {last_loss}", flush=True)
                            running_loss = 0
                            writer.add_scalar("averaged_training_loss", last_loss, step)
                            writer.add_scalar("other_lr", bass_optimizer_scheduler.get_last_lr()[0], step)
                            writer.add_scalar("pretrained_lr", roberta_optimizer_scheduler.get_last_lr()[0], step)

                        if step % checkpoint_step_size == 0:
                            try:
                                step_time = time.time()
                                save_model(model, checkpoint_dir / f"{step}.pth")
                                print(f"checkpoint at steps = {step} has been created after {step_time - start_time} seconds")
                                if step_limit != math.inf:
                                    remaining_time = (step_limit - step)/checkpoint_step_size * (step_time - last_step_time)
                                    print(f"step limit will be reached approx. at {(datetime.now() + timedelta(seconds=remaining_time)).ctime()}")
                            except:
                                print(f"saving the model at step = {step} failed...", flush=True)
                            finally:
                                last_step_time = step_time

                        if step >= step_limit:
                            print(f"step limit reached at {step_limit}")
                            raise TrainingLimitReached()

                        if time.time() - start_time > time_limit:
                            print(f"time limit reached at {time_limit}")
                            raise TrainingLimitReached()

                    writer.flush()
                        
                end_time_epoch = time.time()
                print(f"epoch finished in {end_time_epoch - start_time_epoch} seconds...", flush=True)
                        
        except TrainingLimitReached:
            pass
    
        end_time = time.time()
        print(f"finished training after {end_time - start_time} seconds...", flush=True)
        
        if split == "train":
            writer.close()

    except Exception:
        rich.traceback.Console().print_exception()
    finally:
        pass

   
if __name__ == '__main__':
    main()



