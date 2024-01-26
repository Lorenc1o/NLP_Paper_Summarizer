from summarizer import Summarizer
from rouge_score import rouge_scorer
from tqdm import tqdm
import torch
from nltk import sent_tokenize
from transformers import DistilBertTokenizer
import json

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

test_location = "preprocessing/data/arxiv_summarization/stories/test.pt"
# Load the test data
test_data = torch.load(test_location)

model_orig = Summarizer()

summaries = {}
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'])
scores = {}
scores['rouge1'] = []
scores['rouge2'] = []
scores['rougeLsum'] = []

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

method = 'sum_of_sums'
ch_sum_len = 3

for key in tqdm(test_data.keys()):
    text = test_data[key]['article']
    abstract = test_data[key]['abstract']
    sum_len = len(abstract)
    abstract = ' '.join(abstract)

    if method == 'sum_of_sums':
        chunks = split_into_chunks(tokenizer, text)
        sums = []

        for j, chunk in enumerate(chunks):
            ch_text = " ".join(chunk)
            summary = model_orig(ch_text, num_sentences=ch_sum_len)
            sums.append(summary)

        summary = " ".join(sums)
        summary = model_orig(summary, num_sentences=sum_len)

    elif method == 'all':
        summary = model_orig(text, num_sentences=sum_len)
    
    summaries[key] = {}
    summaries[key]['real'] = abstract
    summaries[key]['predicted'] = summary
    summaries[key]['rouge_score'] = scorer.score(abstract, summaries[key]['predicted'])
    scores['rouge1'].append(summaries[key]['rouge_score']['rouge1'].fmeasure)
    scores['rouge2'].append(summaries[key]['rouge_score']['rouge2'].fmeasure)
    scores['rougeLsum'].append(summaries[key]['rouge_score']['rougeLsum'].fmeasure)

summaries['rouge1'] = sum(scores['rouge1']) / len(scores['rouge1'])
summaries['rouge2'] = sum(scores['rouge2']) / len(scores['rouge2'])
summaries['rougeLsum'] = sum(scores['rougeLsum']) / len(scores['rougeLsum'])

with open("summaries.json", "w") as outfile: 
    json.dump(summaries, outfile)

print(summaries['rouge1'])
print(summaries['rouge2'])
print(summaries['rougeLsum'])

