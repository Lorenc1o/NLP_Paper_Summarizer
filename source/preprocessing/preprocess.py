from datasets import load_dataset
import os
import subprocess
import argparse
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import torch.nn as nn

import nltk
from nltk import sent_tokenize
nltk.download('punkt')

from rouge_score import rouge_scorer

import numpy as np

import json

def generate_oracle_summary(document, abstract, max_sentences=None, min_sentence_length=20):
    '''
        Generates an oracle summary for a given text with a given summary.

        As in the paper, the oracle summary is the set of sentences that maximizes ROUGE when added one at a time, greedily.

        Args:
            document: the text to summarize
            abstract: the summary of the text
            max_sentences: the maximum number of sentences to include in the summary
            min_sentence_length: the minimum length of a sentence to be included in the summary

        Returns:
            oracle_summary: a list of sentences that make up the oracle summary
    '''
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    sentences = sent_tokenize(document)
    
    # Initialize variables
    selected_sentences = set()
    best_rouge = 0.0
    oracle_summary = []

    # Iterate over sentences and select the one that maximizes ROUGE when added
    while True:
        best_sentence = None
        for i, sentence in enumerate(sentences):
            if i in selected_sentences or len(sentence) < min_sentence_length:
                continue

            # Create a temporary summary with the current sentence added
            temp_summary = ' '.join([sentences[j] for j in selected_sentences | {i}])

            # Calculate ROUGE score
            scores = scorer.score(abstract, temp_summary)
            rouge_score = np.mean([scores['rouge1'].fmeasure, scores['rougeL'].fmeasure])

            # Check if this is the best score so far
            if rouge_score > best_rouge:
                best_rouge = rouge_score
                best_sentence = i

        # Break if no improvement or max_sentences reached
        if best_sentence is None or (max_sentences and len(selected_sentences) >= max_sentences):
            break

        # Add the best sentence to the selected set
        selected_sentences.add(best_sentence)

    # Construct the oracle summary
    oracle_summary = [sentences[i] for i in sorted(list(selected_sentences))]
    return oracle_summary

def split_into_chunks(tokenizer, text, max_n_tokens = 512, min_n_tokens = 200):
    """Splitting the Article into a set sentence that comprise a max of 512 tokens
    min_n_tokens - to remove too small chunks that are left in the end. """
    # text = data['article']
    
    sentences = sent_tokenize(text)
    chunks = [[]]
    n_tokens = 0

    for sentence in sentences:

        encoded_dict = tokenizer.encode_plus(
            sentence,
            add_special_tokens=False,  # Since [CLS] and [DEL] are already added in sentences
            return_attention_mask=True,
            return_tensors='pt',
            truncation=False)

        # Extract input_ids and attention_mask
        input_ids = encoded_dict['input_ids'][0]

        n_tokens += len(input_ids) + 2
        if n_tokens < max_n_tokens:
            chunks[-1].append(sentence)
        else:
            chunks.append([])
            n_tokens = 0

    new_chunks = []
    for chunk in chunks:
        encoded_dict = tokenizer.encode_plus(
            " ".join(chunk),
            add_special_tokens=False,  # Since [CLS] and [DEL] are already added in sentences
            return_attention_mask=True,
            return_tensors='pt',
            truncation=False)

        # Extract input_ids and attention_mask
        input_ids = encoded_dict['input_ids'][0]
        # print('length:',len(input_ids))

        if len(input_ids) >= 512:
            raise Exception("Number of tokens exceeded 512!!!")
        elif len(input_ids) >= min_n_tokens:
            # last chunks are usually very short, skip them
            new_chunks.append(chunk)    
       
    return new_chunks

def zip_to_stories(tokenizer, zip_dir, stories_dir, min_sentence_length, ch_sum_sent):
    '''
        Load the raw data from the zip files and save it to the stories as a json file

        Args:
            zip_dir: the directory containing the zip files
            stories_dir: the directory to save the raw data to
            min_sentence_length: the minimum length of a sentence to be included in the summary

        Returns:
            None
    '''
    dataset = load_dataset(zip_dir+"arxiv_summarization.py", 'section')

    for dataset_split in ['train', 'validation', 'test']:
        data_json = {}
        for i, data in enumerate(dataset[dataset_split]):
            # split into chunks
            chunks = split_into_chunks(tokenizer, data['article'])
            for j, chunk in enumerate(chunks):
                text = " ".join(chunk)
                data_json[dataset_split + '_' + str(i) + '_' + str(j)] = {
                    'article': text,
                    'abstract': generate_oracle_summary(text, data['abstract'], ch_sum_sent, min_sentence_length)
                }

            # Print every 10 articles
            if (i + 1) % 10 == 0:
                print(f"{i + 1} articles processed in {dataset_split} dataset")

            # Stop when there are 300 articles for train
            if dataset_split == 'train' and i == 10000:
                break
            # 100 articles for validation and test
            elif dataset_split != 'train' and i == 1000:
                break

        # Save as json file
        with open(stories_dir + f'{dataset_split}.json', 'w') as f:
            json.dump(data_json, f)

def preprocess(text):
    '''
        Preprocess the text by tokenizing it into sentences and adding special tokens
        
        Args:
            text: the text to preprocess
            
        Returns:
            tokenized_text: the tokenized text in sentence form
    '''
    sentences = sent_tokenize(text)
    tokenized_text = ['[CLS] ' + sent + ' [SEP]' for sent in sentences]
    return tokenized_text

def encode_sentences(sentences, tokenizer, max_length=512):
    '''
    Encode the concatenated sentences using the tokenizer. Each sentence has [CLS] at the start and [DEL] at the end.

    Args:
        sentences: the sentences to encode
        tokenizer: the tokenizer to use
        max_length: the maximum length of the sequence after concatenation

    Returns:
        input_ids: the input ids for the concatenated sentences
        attention_masks: the attention masks for the concatenated sentences
        cls_idx: mask for the [CLS] tokens
    '''
    # Concatenate all sentences into one large string
    full_text = ' '.join(sentences)
    encoded_dict = tokenizer.encode_plus(
        full_text,
        add_special_tokens=False,  # Since [CLS] and [DEL] are already added in sentences
        return_attention_mask=True,
        return_tensors='pt',
        truncation=False,
        max_length=max_length
    )

    # Extract input_ids and attention_mask
    input_ids = encoded_dict['input_ids'][0]
    attention_masks = encoded_dict['attention_mask'][0]

    # Find indices of [CLS] tokens
    cls_token_id = tokenizer.cls_token_id
    cls_idx = torch.zeros(len(input_ids), dtype=torch.bool)
    for i, token_id in enumerate(input_ids):
        if token_id == cls_token_id:
            cls_idx[i] = True

    print("Number of sentences:", len(sentences))
    print("Number of tokens:", len(input_ids))
    print("Number of [CLS] tokens:", sum(cls_idx))

    return input_ids, attention_masks, cls_idx

def pipeline(text, tokenizer):
    sentences = preprocess(text)
    input_ids, attention_masks, cls_idx = encode_sentences(sentences, tokenizer)
    return sentences, input_ids, attention_masks, cls_idx

def pipeline_json(json_file, tokenizer):
    '''
        Preprocess the json file and save the processed version in .pt format (suitable for loading with torch.load())

        Args:
            json_file: the json file to preprocess
            tokenizer: the tokenizer to use
            ea_embedding: the embedding for even sentences
            eb_embedding: the embedding for odd sentences

        Returns:
            None
    '''
    with open(json_file) as f:
        data = json.load(f)

    for key in data:
        print("Processing", key)
        sentences, input_ids, attention_masks, cls_idx = pipeline(data[key]['article'], tokenizer)
        data[key]['input_ids'] = input_ids.tolist()
        data[key]['attention_masks'] = attention_masks.tolist()
        data[key]['cls_idx'] = cls_idx.tolist()
        # Transform the abstract from list of sentences to a {0,1} vector, where 1 indicates that the sentence i in the article is in the abstract
        abstract = data[key]['abstract']
        abstract_vector = [0] * len(sentences)
        for sentence in abstract:
            sentence = preprocess(sentence)[0]
            if sentence in sentences:
                abstract_vector[sentences.index(sentence)] = 1
        data[key]['abstract_vector'] = abstract_vector
        print("Done processing", key)

    pt_file = json_file[:-5] + '.pt'
    torch.save(data, pt_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_dir", default="data/arxiv_summarization/", type=str, help="directory containing the zip files")
    parser.add_argument("--zip_to_stories", default=True, type=bool, help="whether to convert the zip files to stories")
    parser.add_argument("--ch_sum_sent", default=3, type=int, help="number of sentences for summaries of 512-token chunks")
    parser.add_argument("--stories_dir", default="data/arxiv_summarization/stories/", type=str, help="directory to save the raw data to")
    parser.add_argument("--json_dir", default="data/arxiv_summarization/stories/", type=str, help="directory to save the json files to")
    parser.add_argument("--max_sentences", default=3, type=int, help="maximum number of sentences to include in the summary")
    parser.add_argument("--min_sentence_length", default=10, type=int, help="minimum length of a sentence to be included in the summary")
    parser.add_argument("--model_name", default="distilbert-base-uncased", type=str, help="name of the model to use")
    parser.add_argument("--output_dir", default="data/arxiv_summarization/summaries/", type=str, help="directory to save the summaries to")
    parser.add_argument("--preprocess", default=True, type=bool, help="whether to preprocess the data")
    args = parser.parse_args()

    tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)

    if args.zip_to_stories:
        zip_to_stories(tokenizer, args.zip_dir, args.stories_dir, args.min_sentence_length, args.ch_sum_sent)

    if args.preprocess:
        # Preprocess the data
        pipeline_json(args.json_dir + 'train.json', tokenizer)
        pipeline_json(args.json_dir + 'validation.json', tokenizer)
        pipeline_json(args.json_dir + 'test.json', tokenizer)

