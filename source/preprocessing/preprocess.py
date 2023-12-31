from datasets import load_dataset

# Assuming your dataset class is in a file named 'arxiv_dataset.py'
# and you have already placed your data files in the expected directory.

# Load the dataset
dataset = load_dataset('data/arxiv_summarization.py', 'section')  # Or 'document', depending on the config you want to use

# Accessing the train, validation, and test splits
train_dataset = dataset['train']
val_dataset = dataset['validation']
test_dataset = dataset['test']

print(train_dataset[0])
