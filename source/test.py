from rouge_score import rouge_scorer
from models.models import Summarizer
import torch
import json
import argparse
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='output/model.pt', type=str, help='path to the model')
    parser.add_argument('--test_data', default='preprocessing/data/dialogueSum/stories/test.pt', type=str, help='path to the test data')
    parser.add_argument('--output_file', default='output/summaries.json', type=str, help='path to the output file')
    parser.add_argument('--sum_len', default=3, type=int, help='number of sentences to include in the summary')
    parser.add_argument('--model_type', default='transformer', type=str, help='type of model to use: transformer or linear')
    args = parser.parse_args()

    # Load the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Summarizer(device, args.model_type)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model.to(device)

    # Load the test data
    with open(args.test_data, 'rb') as f:
        data = torch.load(f)

    # Predict the summaries
    summaries = {}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'])
    scores = {}
    scores['rouge1'] = []
    scores['rouge2'] = []
    scores['rougeLsum'] = []
    for key in tqdm(data):
        text = data[key]['article']
        abstract = data[key]['abstract']
        abstract = ' '.join(abstract)
        summary, _ = model.predict(text, args.sum_len)
        summaries[key] = {}
        summaries[key]['real'] = abstract
        summaries[key]['predicted'] = summary
        summaries[key]['rouge_score'] = scorer.score(abstract, summaries[key]['predicted'])
        scores['rouge1'].append(summaries[key]['rouge_score']['rouge1'].fmeasure)
        scores['rouge2'].append(summaries[key]['rouge_score']['rouge2'].fmeasure)
        scores['rougeLsum'].append(summaries[key]['rouge_score']['rougeLsum'].fmeasure)

    print('Average rouge1 score:', sum(scores['rouge1']) / len(scores['rouge1']))
    print('Average rouge2 score:', sum(scores['rouge2']) / len(scores['rouge2']))
    print('Average rougeLsum score:', sum(scores['rougeLsum']) / len(scores['rougeLsum']))
    summaries['rouge1'] = sum(scores['rouge1']) / len(scores['rouge1'])
    summaries['rouge2'] = sum(scores['rouge2']) / len(scores['rouge2'])
    summaries['rougeLsum'] = sum(scores['rougeLsum']) / len(scores['rougeLsum'])    

    # Save the summaries
    with open(args.output_file, 'w') as f:
        json.dump(summaries, f)

