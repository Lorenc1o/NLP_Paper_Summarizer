import torch
from models.models import EncoderModel, LinearClassifier, Summarizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW
import matplotlib.pyplot as plt

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
    if input_ids.shape[0] > max_len:
        input_ids = input_ids[:max_len]
        attention_mask = attention_mask[:max_len]
        # For cls_idx, we only keep the cls_idx that are within the max_len
        cls_idx_upd = cls_idx[cls_idx < max_len]
        # For abstract_vector, we keep the same entries as cls_idx
        abstract_vector = abstract_vector[cls_idx < max_len]
    return input_ids, attention_mask, abstract_vector, cls_idx_upd

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

def collate_fn(batch):
    input_ids, attention_masks, labels, cls_idx = zip(*batch)

    # Padding sequences to the maximum length in this batch
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100) 

    return input_ids_padded, attention_masks_padded, labels_padded, cls_idx

def train(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for input_ids, attention_masks, labels, cls_idx in data_loader:
        optimizer.zero_grad()

        # Ensure the data is on the correct device (CPU or GPU)
        #input_ids = input_ids.to(model.device)
        #attention_masks = attention_masks.to(model.device)
        #labels = labels.to(model.device)
        #if isinstance(cls_idx, tuple):
        #    cls_idx = torch.stack(cls_idx).to(model.device)
        #else:
        #    cls_idx = cls_idx.to(model.device)

        logits = model(input_ids, attention_masks, cls_idx)
        logits = torch.squeeze(logits)

        # Convert labels to float
        labels_float = labels.float()

        loss = criterion(logits, labels_float)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)


def validate(model, data_loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for input_ids, attention_masks, labels, cls_idx in data_loader:
            # Ensure the data is on the correct device (CPU or GPU)
            #input_ids = input_ids.to(model.device)
            #attention_masks = attention_masks.to(model.device)
            #labels = labels.to(model.device)
            #if isinstance(cls_idx, tuple):
            #    cls_idx = torch.stack(cls_idx).to(model.device)
            #else:
            #    cls_idx = cls_idx.to(model.device)

            logits = model(input_ids, attention_masks, cls_idx)
            logits = torch.squeeze(logits)

            # Convert labels to float
            labels_float = labels.float()

            loss = criterion(logits, labels_float)
            total_loss += loss.item()

    return total_loss / len(data_loader)

def plot_loss(train_history, val_history):
    plt.plot(train_history, label='train')
    plt.plot(val_history, label='val')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    training_loc = 'preprocessing/data/arxiv_summarization/stories/test.pt'
    validation_loc = 'preprocessing/data/arxiv_summarization/stories/validation.pt'

    data_train = read_pt_file(training_loc)
    dataset_train = SummarizationDataset(data_train)

    data_val = read_pt_file(validation_loc)
    dataset_val = SummarizationDataset(data_val)

    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset_val, batch_size=32, shuffle=True, collate_fn=collate_fn)

    bert = EncoderModel()
    classifier = LinearClassifier(768, 1)
    model = Summarizer(bert, classifier, 'cuda:0')

    optimizer = AdamW(model.parameters(), lr=1e-5)

    criterion = torch.nn.BCELoss()

    train_history = []
    val_history = []

    for epoch in range(10):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

    torch.save(model.state_dict(), 'model.pt')

    plot_loss(train_history, val_history)
  