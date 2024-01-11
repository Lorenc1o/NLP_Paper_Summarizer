import torch
from models.models import EncoderModel, LinearClassifier, TransformerClassifier, Summarizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW
import matplotlib.pyplot as plt
import argparse

def read_pt_file(path):
    '''
        Read a .pt file and return its contents

        Args:
            path: the path to the .pt file

        Returns:
            the contents of the .pt file
    '''
    return torch.load(path)

# For now we trim the sequences to 512 tokens
def trim_sequences(input_ids, attention_mask, abstract_vector, cls_idx, max_len=512):
    i = 0
    if input_ids.shape[0] > max_len:
        i += 1
        input_ids = input_ids[:max_len]
        attention_mask = attention_mask[:max_len]
        cls_idx = cls_idx[:max_len]
        abstract_vector = abstract_vector[:max_len]
    
    return input_ids, attention_mask, abstract_vector, cls_idx

class SummarizationDataset(Dataset):
    def __init__(self, data):
        super(SummarizationDataset, self).__init__()
        self.data = list(data.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids, attention_masks, abstract_vector, cls_idx = trim_sequences(torch.tensor(item['input_ids']), torch.tensor(item['attention_masks']), torch.tensor(item['abstract_vector']), torch.tensor(item['cls_idx']))
        return input_ids, attention_masks, abstract_vector, cls_idx

def labels_to_mask(size, labels):
    '''
        Convert a vector of labels to a mask

        Args:
            size: the size of the mask
            labels: a vector of labels

        Returns:
            a mask
    '''
    mask = torch.zeros(size)
    for i, label in enumerate(labels):
        if label == 1:
            mask[i] = 1
    return mask

def bool_to_mask(bools):
    '''
        Convert a boolean tensor to a mask

        Args:
            bools: a boolean tensor

        Returns:
            a mask
    '''
    mask = bools.long()
    return mask

def collate_fn(batch):
    input_ids, attention_masks, labels, cls_idx = zip(*batch)

    # Convert labels to masks
    labels = [labels_to_mask(len(input_ids[i]), labels[i]) for i in range(len(labels))]

    # Padding sequences to the maximum length in this batch
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0) 
    cls_idx = bool_to_mask(pad_sequence(cls_idx, batch_first=True, padding_value=0))

    return input_ids_padded, attention_masks_padded, labels_padded, cls_idx

def extract_labels(labels, cls_idx):
    # labels shape: [batch_size, num_tokens]
    # cls_idx shape: [batch_size, num_tokens]
    # 1. Add 1 to all labels: 0 -> 1, 1 -> 2
    labels = labels + 1
    # 2. Multiply labels by cls_idx: 0 and cls_idx -> 1, 1 and cls_idx -> 2, else -> 0
    labels = labels * cls_idx

    max_num_cls = max(cls_idx.sum(dim=1))
    # 3. Pad the labels to the same length
    padded_labels = []
    for i in range(labels.size(0)):
        cls_labels = labels[i,:]
        non_zero_labels = cls_labels[cls_labels != 0]
        num_padding = max_num_cls - non_zero_labels.size(0)
        padding = torch.ones(num_padding, device=non_zero_labels.device)
        padded_labels.append(torch.cat([non_zero_labels, padding], dim=0))
    padded_labels = torch.stack(padded_labels, dim=0)
    # 4. Subtract 1 from all labels: 1 -> 0, 2 -> 1
    padded_labels = padded_labels - 1
    return padded_labels

def train(model, data_loader, optimizer, criterion, verbose=False):
    model.train()
    total_loss = 0

    for input_ids, attention_masks, labels, cls_idx in data_loader:
        optimizer.zero_grad()

        # Ensure the data is on the correct device (CPU or GPU)
        input_ids = input_ids.to(model.device)
        attention_masks = attention_masks.to(model.device)
        cls_idx = cls_idx.to(model.device)
        labels = labels.to(model.device)

        logits = model(input_ids, attention_masks, cls_idx)
        logits = torch.squeeze(logits)

        labels = extract_labels(labels, cls_idx)

        # Convert labels to float
        labels_float = labels.float().squeeze().to(model.device)

        loss = criterion(logits, labels_float)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if verbose:
            print(f'Batch loss: {loss.item()}')

    return total_loss / len(data_loader)


def validate(model, data_loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for input_ids, attention_masks, labels, cls_idx in data_loader:
            # Ensure the data is on the correct device (CPU or GPU)
            input_ids = input_ids.to(model.device)
            attention_masks = attention_masks.to(model.device)
            cls_idx = cls_idx.to(model.device)
            labels = labels.to(model.device)

            logits = model(input_ids, attention_masks, cls_idx)
            logits = torch.squeeze(logits)

            labels = extract_labels(labels, cls_idx)

            # Convert labels to float
            labels_float = labels.float().squeeze().to(model.device)

            loss = criterion(logits, labels_float)
            total_loss += loss.item()

    return total_loss / len(data_loader)

def plot_loss(train_history, val_history):
    plt.plot(train_history, label='train')
    plt.plot(val_history, label='val')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_loc", default='preprocessing/data/dialogueSum/stories/train.pt', type=str, help="path to the .pt training data")
    parser.add_argument("--valid_loc", default='preprocessing/data/dialogueSum/stories/validation.pt', type=str, help="path to the .pt validation data")
    parser.add_argument("--model_loc", default='model.pt', type=str, help="path to the model")
    parser.add_argument("--output_dir", default='output/', type=str, help="directory to save the model and the training history")
    parser.add_argument("--model_type", default='transformer', type=str, help="type of model to use: transformer or linear")
    parser.add_argument("--verbose", default=False, type=bool, help="whether to print the loss after each batch")
    parser.add_argument("--batch_size", default=10, type=int, help="batch size")
    parser.add_argument("--train_size", default='all', type=str, help="number of examples to train on")
    args = parser.parse_args()

    data_train = read_pt_file(args.train_loc)
    dataset_train = SummarizationDataset(data_train)
    if args.train_size != 'all':
        dataset_train = torch.utils.data.Subset(dataset_train, range(int(args.train_size)))

    data_val = read_pt_file(args.valid_loc)
    dataset_val = SummarizationDataset(data_val)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    bert = EncoderModel()

    if args.model_type == 'transformer':
        classifier = TransformerClassifier(768, 1, 1, 768, 0.1)
    else:
        classifier = LinearClassifier(768, 1)

    device = 'cuda'

    model = Summarizer(bert, classifier, device, args.model_type)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    criterion = torch.nn.BCELoss()

    train_history = []
    val_history = []

    for epoch in range(10):
        train_loss = train(model, train_loader, optimizer, criterion, args.verbose)
        val_loss = validate(model, val_loader, criterion)

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

    torch.save(model.state_dict(), args.output_dir + args.model_loc[:-3] + args.model_type + '.pt')
    torch.save(train_history, args.output_dir + 'train_history' + args.model_type + '.pt')
    torch.save(val_history, args.output_dir + 'val_history.pt' + args.model_type + '.pt')

    plot_loss(train_history, val_history)
  