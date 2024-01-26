# NLP_Paper_Summarizer
Repository for the Paper Summarizer project for the course 'Machine Learning' @ Université Paris-Saclay, CentraleSupélec. BDMA, Fall 2023.

Authors:
- [Jose Antonio Lorencio Abril](https://github.com/Lorenc1o)
- [Sayyor Yusupov](https://github.com/SYusupov)

## Instructions

### Datasets

The datasets can be found at:

- Arxiv-Summarization dataset: download from [Arxiv-Summarization at HuggingFace](https://huggingface.co/datasets/ccdv/arxiv-summarization)
- DialogSum dataset: download from [DialogSum at HuggingFace](https://huggingface.co/datasets/knkarthick/dialogsum)

### Environment

Install the dependencies from the `requirements.txt` file. We recommend using a virtual environment and used Python 3.9.18.

### Preprocessing

To preprocess the datasets, you have to use the `source/preprocessing/preprocess.py` script. You can run it with the following command:

```bash
python source/preprocessing/preprocess.py --zip_dir <zip_location> --zip_to_stories [True/False] --toChunk [True/False] --ch_sum_sent <n_sents> --stories_dir <stories_location> --json_dir <json_location> --max_sentences <max_sents> --min_sentence_length <min_length> --model_name <model_name> --output_dir <output_dir> --preprocess [True/False]
```
Where:
- `zip_dir` is the location of the zip file containing the datasetç
- `zip_to_stories` is a boolean indicating whether to convert the zip files to stories or not
- `toChunk` is a boolean indicating whether to chunk stories longer than 512 tokens or not
- `ch_sum_sent` is the number of sentences used to summarize each chunk
- `stories_dir` is the location of the stories folder
- `json_dir` is the location of the json file to save the preprocessed data
- `max_sentences` is the maximum number of sentences to keep in the whole summary
- `min_sentence_length` is the minimum number of tokens in a sentence to keep it in the summary
- `model_name` is the name of the model used to tokenize the sentences
- `output_dir` is the location of the output folder, to save the summaries
- `preprocess` is a boolean indicating whether to preprocess the data or not

If you use DialogSum, you might need to use the alternative script `source/preprocessing/preprocess_dialogue.py` instead, which works similarly.

### Training

To train the model, you have to use the `source/train.py` script. You can run it with the following command:

```bash
python source/train.py --train_loc <train_dataset_loc> --valid_loc <validation_dataset_loc> --model_loc <path_to_model> --output_dir <output_dir> --model_type <model_type> --verbose [True/False] --batch_size <bsize> --train_size <tsize> --valid_size <vsize> --epochs <n_epochs>
```
Where:
- `train_loc` is the location of the training dataset
- `valid_loc` is the location of the validation dataset
- `model_loc` is the name of the model to use
- `output_dir` is the location of the output folder, to save the model and the logs
- `model_type` is the type of model to use, 'linear' or 'transformer'
- `verbose` is a boolean indicating whether to print the logs or not
- `batch_size` is the batch size to use
- `train_size` is the number of training samples to use
- `valid_size` is the number of validation samples to use
- `epochs` is the number of epochs to train the model

### Evaluation
You can use the script `experiment.sh` to run experiments. You can run different experiments with the script `run_experiments.sh`, where you can add all the different parameters you want to test.

