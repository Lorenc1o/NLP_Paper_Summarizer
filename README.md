# NLP_Paper_Summarizer
Repository for the Paper Summarizer project for the course 'Machine Learning' @ Université Paris-Saclay, CentraleSupélec. BDMA, Fall 2023.

## Instructions

### Preprocessing

Download the dataset from [here](https://huggingface.co/datasets/ccdv/arxiv-summarization) and unzip it in the `source/preprocessing/data` folder.

You need to install the (Stanford CoreNLP)[https://stanfordnlp.github.io/CoreNLP/download.html] package. 

You need to compile it:

```bash
cd /path/to/stanford-corenlp
mvn compile
```

Then, you need to add the following environment variable:

```bash
export CLASSPATH=/path/to/stanford-corenlp/*
```

Then, you can run the preprocessing script. The following command works from the root of the repository:

```bash
python source/preprocessing/preprocess.py --preprocess true --tokenize true
```
If you want to only preprocess/tokenize, just don't pass the `--preprocess` or `--tokenize` flags.
If you want to use different input/output folders, you can use the `--raw_path` (the data from the dataset), `--save_path` (the preprocessed data) and `--tokenized_path` (the tokenized data) flags. By default, the script will look for the data in the `source/preprocessing/data` folder and save the preprocessed/tokenized data in the `source/preprocessing/data/processed` and `source/preprocessing/data/tokenized` folders respectively.
