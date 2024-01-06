import argparse
from transformers import DistilBertTokenizer

import nltk
nltk.download('punkt')

import pandas as pd
import json

from preprocess import generate_oracle_summary, pipeline_json

def zip_to_stories(zip_dir, stories_dir, min_sentence_length=20):
    '''
        Load the raw data from the zip files and save it to the stories as a json file

        Args:
            zip_dir: the directory containing the zip files
            stories_dir: the directory to save the raw data to
            min_sentence_length: the minimum length of a sentence to be included in the summary

        Returns:
            None
    '''
    train_data = pd.read_csv(zip_dir+"train.csv")
    validation_data = pd.read_csv(zip_dir+"validation.csv")
    test_data = pd.read_csv(zip_dir+"test.csv")

    datasets = {'train': train_data, 'validation': validation_data, 'test': test_data}

    for dataset_split, data in datasets.items():
        data_json = {}
        for i, row in data.iterrows():
            data_json[dataset_split + '_' + str(i)] = {
                'article': row['dialogue'],
                'abstract': generate_oracle_summary(row['dialogue'], row['summary'], 5, min_sentence_length)
            }

            # Print every 10 articles
            if (i + 1) % 10 == 0:
                print(f"{i + 1} articles processed in {dataset_split} dataset")

            # # Stop when there are 300 articles for train
            # if dataset_split == 'train' and i == 30:
            #     break
            # # 100 articles for validation and test
            # elif dataset_split != 'train' and i == 2:
            #     break

        # Save as json file
        with open(stories_dir + f'{dataset_split}.json', 'w') as f:
            json.dump(data_json, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_dir", default="data/dialogueSum/", type=str, help="directory containing the zip files")
    parser.add_argument("--zip_to_stories", default=True, type=bool, help="whether to convert the zip files to stories")
    parser.add_argument("--stories_dir", default="data/dialogueSum/stories/", type=str, help="directory to save the raw data to")
    parser.add_argument("--json_dir", default="data/dialogueSum/stories/", type=str, help="directory to save the json files to")
    parser.add_argument("--max_sentences", default=3, type=int, help="maximum number of sentences to include in the summary")
    parser.add_argument("--min_sentence_length", default=10, type=int, help="minimum length of a sentence to be included in the summary")
    parser.add_argument("--model_name", default="distilbert-base-uncased", type=str, help="name of the model to use")
    parser.add_argument("--output_dir", default="data/dialogueSum/summaries/", type=str, help="directory to save the summaries to")
    parser.add_argument("--preprocess", default=True, type=bool, help="whether to preprocess the data")
    args = parser.parse_args()

    if args.zip_to_stories:
        zip_to_stories(args.zip_dir, args.stories_dir, args.min_sentence_length)

    if args.preprocess:
        tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)

        # Preprocess the data
        pipeline_json(args.json_dir + 'train.json', tokenizer)
        pipeline_json(args.json_dir + 'validation.json', tokenizer)
        pipeline_json(args.json_dir + 'test.json', tokenizer)

