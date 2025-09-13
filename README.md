# Multilingual GPT-BERT Baseline for BabyBabelLM (Jumelet et al f.c.)

https://huggingface.co/BabyLM-community 

As part of the BabyBabelLM (Multilingual BabyLM) team, we develop Multilingual BabyLM corpora for several languages. 

This is a GPT-BERT multilingual baseline architecture for these datasets. 

I modify the pretraining code provided by Charpentier et al (2024)

## Setup

```
python3 -m venv venvs/demo; source venvs/demo/bin/activate
hf auth login
```

## Training

There are three default modes for training: monolingual GPT-BERT trained on one BabyBabelLM corpus, a small multilingual model (trained on Tier 1 multilingual corpora) and one large multilingual model.  

The script automates the following workflow:

Sets up the job environment and logs job details.

Creates and activates a Python virtual environment with the necessary dependencies.

Builds the tokenizer.

Preprocesses the dataset for the specified language based on specification (multilingual large (Tiers 1,2,3) or small (tier 1 ONLY) or monolingual). 

Trains the GPT-BERT model using the preprocessed data.


The script requests the following resources (Cambridge HPC Specific) – please modify for your own GPU setup! 
```
Job name: baby-lm-mono
Account: BUTTERY-SL2-GPU
Nodes: 1
Tasks: 1
GPU partition: ampere
Wall time: 12:00:00
Failure notification: email on job failure
```

You can adjust these parameters (#SBATCH directives) based on your cluster’s configuration.

### Monolingual GPT-BERTs
```
cd scripts
sbatch run_mono.slurm zho
sbatch run_mono.slurm nld
sbatch run_mono.slurm deu
```
# Multilingual GPT-BERTs
```
cd scripts
sbatch run_multiall.slurm
```


```
cd scripts
sbatch run_multismall.slurm
```




## Training

 ## Tokenizer

Runs tokenizers/tokenizer.py to build the tokenizer called ./tokenizer/tokenizer.json.

## Preprocessing

Runs preprocess/updated_preprocess.py with the following settings:

Dataset type
Language (monolingual only): $LANG_CODE
Sequence length: 128
Shard size: 100 MB
Batch size: 1000

Output is stored in 
data/MONOLINGUAL/$LANG_CODE/train OR valid/
data/MULTILINGUAL-ALL/train OR valid/ 
data/MULTILINGUAL-SMALL/train OR valid/ 

## Training

Runs pretraining/train_single_gpu.py with:

Tokenizer path: ../tokenizers/tokenizer.json
Config: ../configs/small.json
Batch sizes: local = 32, global = 256
Precision: mixed (FP16/FP32)

## BabyLM Tokenization & Sharding Pipeline

This repository provides scripts to tokenize multilingual BabyLM datasets, save them as PyTorch-compatible tensors, and split them into manageable shards for training.

1. **Encoding and Conversion**

   The script `encode_and_convert.py` processes the datasets in two steps:

   1. **Streaming uint16 `.bin`**  
      All token IDs are concatenated and written to a `.bin` file using `save_bin_stream`.  
      Example outputs:
      - `../data/babybabellm_all.bin` (train)  
      - `../data/dev_babybabellm.bin` (validation)

   2. **Convert to Torch tensor**  
      The uint16 `.bin` files are loaded into a single NumPy array and converted into a single `torch.LongTensor` using `convert_uint16_bin_to_torch`.  
      Example outputs:
      - `../data/babybabellm_all_torch.bin` (train)  
      - `../data/dev_babybabellm_torch.bin` (validation)

   ⚠️ Note: These are single tensors containing all tokens concatenated, **not a list of documents**.

2. **Sharding**

The preprocess scripts `split_dataset.py` (modified for single tensors) splits the huge tensor into smaller shards suitable for training. This creates shards in  `../data/shards/train/` and  `../data/shards/valid/`. Each shard is a smaller tensor that can be efficiently loaded during training.
