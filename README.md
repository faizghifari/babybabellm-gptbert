# Multilingual GPT-BERT Baseline for BabyBabelLM (Jumelet et al f.c.)

https://huggingface.co/BabyLM-community 

As part of the BabyBabelLM (Multilingual BabyLM) team, we develop Multilingual BabyLM corpora for several languages. 

This is a GPT-BERT multilingual baseline architecture for these datasets. 

I modify the pretraining code provided by Charpentier et al (2024)

## Setup

```
python3 -m venv venvs/demo; source venvs/demo/bin/activate
pip3 install -r requirements.txt
```

## Multilingual Datasets and Tokenizers

```
cd tokenizers
python3 tokenizer.py
```
