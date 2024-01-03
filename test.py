from transformers import DistilBertTokenizer, DistilBertModel
import torch
import torch.nn as nn

import nltk
from nltk import sent_tokenize
nltk.download('punkt')

from rouge_score import rouge_scorer

import numpy as np

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def preprocess(text):
    sentences = sent_tokenize(text)
    tokenized_text = ['[CLS] ' + sent + ' [SEP]' for sent in sentences]
    return tokenized_text

def initialize_segment_embeddings(model):
    ea_embedding = torch.nn.Embedding(2, model.config.dim)
    eb_embedding = torch.nn.Embedding(2, model.config.dim)
    return ea_embedding, eb_embedding

def encode_sentences(sentences, tokenizer):
    input_ids = []
    attention_masks = []
    segment_ids = []  # Keep track of segments (odd or even)

    for i, sent in enumerate(sentences):
        encoded_dict = tokenizer.encode_plus(
                            sent,
                            add_special_tokens = False,
                            max_length = 128,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt',
                       )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        segment_ids.append(torch.full((1, 128), i % 2))  # 0 for even, 1 for odd sentences
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    segment_ids = torch.cat(segment_ids, dim=0)
    
    return input_ids, attention_masks, segment_ids

def add_segment_embeddings(embeddings, segment_ids, ea_embedding, eb_embedding):
    # Get the batch size and sequence length from embeddings
    batch_size, seq_length, hidden_size = embeddings.size()

    # Expand segment embeddings to match the dimensions of BERT embeddings
    ea_embeddings = ea_embedding(segment_ids).view(batch_size, seq_length, hidden_size)
    eb_embeddings = eb_embedding(segment_ids).view(batch_size, seq_length, hidden_size)

    # Add segment embeddings to the original embeddings
    enhanced_embeddings = embeddings + torch.where(segment_ids.unsqueeze(-1) == 0, ea_embeddings, eb_embeddings)

    return enhanced_embeddings

def get_embeddings(input_ids, attention_masks, model):
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)
    return outputs.last_hidden_state

def extract_sentence_embeddings(embeddings):
    # Assuming that the first token of each sentence is [CLS]
    return embeddings[:,0,:]

def pipeline(text, model, tokenizer, ea_embedding, eb_embedding):
    sentences = preprocess(text)
    input_ids, attention_masks, segment_ids = encode_sentences(sentences, tokenizer)
    embeddings = get_embeddings(input_ids, attention_masks, model)
    embeddings = add_segment_embeddings(embeddings, segment_ids, ea_embedding, eb_embedding)
    sentence_embeddings = extract_sentence_embeddings(embeddings)
    return sentences, sentence_embeddings

class Classifier(nn.Module):
    '''
    Classifier

    A linear layer with sigmoid activation function.
    '''
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores
    

def generate_oracle_summary(document, abstract, max_sentences=None):
    '''
        Generates an oracle summary for a given text with a given summary.

        As in the paper, the oracle summary is the set of sentences that maximizes ROUGE when added one at a time, greedily.

        Args:
            document: the text to summarize
            abstract: the summary of the text
            max_sentences: the maximum number of sentences to include in the summary

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
            if i in selected_sentences:
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

if __name__ == '__main__':
    doc = 'source/preprocessing/data/processed/val/val_1.story'

    with open(doc, 'r') as f:
        document = f.read()
        document = document.replace('\n', ' ')
        document = document.replace('  ', ' ')
        document = document.lower()

        # the sentences of the summary are those that are preceded by @highlight, there can be multiple
        abstract = document.split('@highlight')[1:]
        abstract = " ".join(abstract).strip()
        document = document.split('@highlight')[0].strip()



    oracle_summary = generate_oracle_summary(document, abstract, max_sentences=3)

    print("Oracle summary:")
    print(oracle_summary)
    print("------------------------")
    print("Abstract:")
    print(abstract) 