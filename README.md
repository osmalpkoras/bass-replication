# A Second Look on BASS - Boosting Abstractive Summarization with Unified Semantic Graphs

 ![Sketch of the BASS Framework](./assets/BASS%20Framework.jpg)

This repository contains the code and resources for the paper titled "A Second Look on BASS - Boosting Abstractive Summarization with Unified Semantic Graphs." BASS is an abstractive summarization framework, which uses i) dependency parse trees to generate Unified Semantic Graphs (USG) for documents to compress and relate information across the input document, and ii) a model architecture, which incorporates the graph information.

This README provides detailed instructions on how to set up, preprocess data, train models, generate summaries, and evaluate model outputs for abstractive summarization tasks using Unified Semantic Graphs (USG). For replicating our experiments, you must follow steps 1-5 and pass command line parameters in step [3. Training](#3.-training) as desired. Simlarily, the command line parameters in step [4. Generation](#4.-generation) and [5. Evaluation](#5.-evaluation) must be adjusted to the generated model checkpoints and outputs. If you directly want to use the [Model Weights](#model-weights), you can skip the training.

## Table of Contents

* [1. Installation](#1.-installation)
* [2. Data Preparation](#2.-data-preparation)
* [3. Training](#3.-training)
* [4. Generation](#4.-generation)
* [5. Evaluation](#5.-evaluation)
* [Model Weights](#model-weights)

## 1. Installation

Before proceeding, please ensure you have installed a Java Development Kit (v8.0.2). You can install the conda environment from one of the `environment.yml` files in this repository. Alternatively, you can manually install the environment following these steps:


1. Create a Conda Environment:

   ```bash
   conda create -n bass python=3.10
   conda activate bass
   ```
2. Install Python Dependencies:

   ```bash
   conda install -c conda-forge pyvis=0.3.1 rich=13.3.1 click=8.1.3 huggingface_hub=0.12.1 numpy=1.23.5
   conda install -c huggingface transformers=4.26.1
   pip install "javaobj-py3==0.4.3" "stanza==1.4.2" "rouge-score==0.1.2" "bert-score==0.3.13" "tensorboard==2.12.1"
   ```
3. Install PyTorch:
   If you have a Mac computer with Apple silicon (M1/M2)

   ```bash
   conda install pytorch::pytorch=2.0.0 torchvision=0.15.0 torchaudio=2.0.0 -c pytorch
   ```

   If you have a CUDA-enabled Windows/Linux computer:

   ```bash 
   conda install pytorch=2.0.0 torchvision=0.15.0 torchaudio=2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
   ```

## Data Preparation

Before running the pre-processing with CoreNLP, you need to download and unzip the [CoreNLP package (version 4.5.2)](https://stanfordnlp.github.io/CoreNLP/history.html) to the `java` directory of this repository. The library should now be placed in `java/stanford-corenlp-4.5.2`.
You will also need to download the [BigPatent dataset](https://evasharma.github.io/bigpatent/). Unzip it to a directory of your choice.
The `Preprocessor.java` script performs preprocessing (and graph construction) on the dataset using CoreNLP. Detailed instructions for running the preprocessor:

* **Compile Preprocessor.java:**

  ```bash
  cd java
  mkdir out
  javac -d out Preprocessor.java ./preprocess/*java ./helpers/*.java ./semanticgraph/*.java ./bass/data/*.java ./bass/model/*.java -classpath "stanford-corenlp-4.5.2/*"
  ```
* **Java Preprocessing:**
  Run the Java preprocessor to run the CoreNLP parser and to construct the original Unified Semantic Graphs USG$_{src}$. Example usage:

  ```bash
  java -cp out:"stanford-corenlp-4.5.2/*" -Xmx8G Preprocessor SOURCE_DIR TEMP_DIR TARGET_DIR 0 10000 100 500 3
  ```
  * `SOURCE_DIR`: The directory containing the extracted BigPatent zip archive.
  * `TEMP_DIR`: A temporary directory for intermediate files.
  * `TARGET_DIR`: The target directory where output files will be copied.
  * `POOL_INDEX`: The index of the worker preprocessing this dataset (0 <= pool_index <= pool_size).
  * `POOL_SIZE`: The number of workers separately preprocessing this dataset.
  * `TIMEOUT`: The time in seconds given to preprocess a chunk.
  * `CHUNK_SIZE`: The number of words per chunk.
  * `MAX_CHUNKS`: The maximum number of chunks to preprocess.


* **Python Preprocessing:**
  Run `run_preprocessing.py` to prepare the dataset for training and generation. Pass the right preprocessor type to prepare the dataset either for the original USG$*{src} (*`USGsrc`) or our paper-compliant replicated USG${ppr}$ (`USGppr`) graphs.

  Example usage:

  ```bash
  python run_preprocessing.py SOURCE_DIR TEMP_DIR OUTPUT_DIR PREPROCESSOR_TYPE --pool-index 0 --pool-size 3000 --worker-count 1 --timeout 10000
  ```
  * `SOURCE_DIR`: The root directory of the source dataset. This should point to the target directory of the Java pre-processing step.
  * `TEMP_DIR`: A temporary directory used for intermediate files.
  * `OUTPUT_DIR`: The root directory into which the preprocessed dataset should be output.
  * `PREPROCESSOR_TYPE`: The kind of preprocessing to perform (either `USGsrc` or `USGppr`).
  * `--pool-index`: The index of the pool worker preprocessing the dataset.
  * `--pool-size`: The total number of pool workers preprocessing the dataset.
  * `--worker-count`: The number of workers to use for parallelization.
  * `--timeout`: The timeout for preprocessing a single datapoint.
  * `--parallelization-type`: The type of parallelization if more than one worker is used.

Please use different target dataset directories for the two graph types.
Customize the paths and options according to your needs.

## Training

To train the model, use `train.py`. Example usage:

```bash
python train.py DATASET_DIR CHECKPOINT_DIR --time-limit 00-00:00:00 --step-limit 300000 --checkpoint-step-size 10000 --device cuda:0,1,2,3,4,5 --batch-size 48 --model Bass
```

* `DATASET_DIR`: The directory containing the training dataset. This should point to an output directory used for the Python pre-processing step.
* `CHECKPOINT_DIR`: The directory to save model checkpoints.
* `--time-limit`: The time limit after which the training should terminate gracefully (e.g., `05-12:30:00` will run for 5 days, 12 hours and 30 minutes).
* `--step-limit`: The step limit after which the training should terminate gracefully.
* `--checkpoint-step-size`: The number of steps to perform before saving another model checkpoint.
* `--device`: The device to train on (e.g., `cuda:0,1,2,3,4,5` for multiple GPUs).
* `--batch-size`: The batch size for a single training step.
* `--accumulate-gradients`: The number of batches to accumulate for each step, effectively changing the batch size.
* `--max-epochs`: The maximum number of epochs to iterate through the training set.
* `--checkpoint`: The specific checkpoint to continue training with, or model weights.
* `--model`: The model type to initialize (e.g., `Bass`, `RTS2S`, `exRTS2S`), required if no prior checkpoint is loaded or when model weights are loaded.

Customize the paths and options according to your needs.

## Generation

To generate summaries using the trained model, use `generate.py`. Example usage:

```bash
python generate.py DATASET_DIR CHECKPOINT_DIR --device cuda:1,2,3,4,5 --batch-size 180 --checkpoint 300000 --output-file output
```

* `DATASET_DIR`: The directory of the dataset that contains the preprocessed test split. This should point to an output directory used for the Python pre-processing step.
* `CHECKPOINT_DIR`: The directory of model checkpoints. This should point to a checkpoint directory used for the training step or to a directory containing model weights.
* `--device`: The device to generate summaries on (e.g., "cuda:1,2,3,4,5").
* `--batch-size`: The batch size for beam search (with beam width = 5, integer).
* `--checkpoint`: The exact checkpoint name to load. This can either be the numerical value at the step a model was saved (durint training) or the name of the file containing model weights. If not given, the script will load the highest numerical file name (integer).
* `--output-file`: The file name ending of the generation output (string).
  Customize the paths and options according to your needs.

## Evaluation

To evaluate the generated summaries, use `evaluate.py`. Example usage:

```bash
python evaluate.py model/weights/Bass_ppr.output model/weights/scores --output-file generated_scores
```

* `model-output-file`: The model output file generated by `generate.py`. This should point to the output file generated in the generation step.
* `evaluation-output-dir`: The directory where scores should be saved.
* `--output-file`: The prefix for the files generated by this script (optional, defaults to the model-output file name).

Customize the paths and options according to your needs.

We perform significance testing using the paired bootstrap resampling method as described in [The Hitchhiker’s Guide to Testing Statistical Significance in Natural Language Processing](https://aclanthology.org/P18-1128) (Dror et al., ACL 2018). To conduct significance testing, you can consult the authors' repository or the respective section of the paper for detailed information.


## Model Weights

You can obtain pre-trained model weights by downloading them from the provided URL: [Download Model Weights](https://drive.google.com/drive/folders/1b-9rmbmU_eu_Czm9uMHeVpdPpxvFo_Pz?usp=sharing). You can place them in `model/weights`.

Once downloaded, you can use these pre-trained weights in both the training and generation steps by specifying the parent directory containing the weights as the checkpoint directory and provide the name of the file as the checkpoint.

## Citation

If you find this work helpful for your research, please consider citing our paper:

```
[include citation here]
```

## License

This repository is released under the [MIT License](LICENSE). You can find the Java source code provided by the original authors in `java/bass`. The copyright of the respective author(s) remains intact.

## Get in Touch

If you have any questions, encounter issues, or wish to collaborate, don't hesitate to reach out to us. We appreciate your interest and feedback.

## Acknowledgments

We thank the authors of [BASS: Boosting Abstractive Summarization with Unified Semantic Graph](https://aclanthology.org/2021.acl-long.472) (Wu et al., ACL-IJCNLP 2021) for their permission to release their Java source code.