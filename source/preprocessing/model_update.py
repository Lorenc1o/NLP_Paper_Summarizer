from transformers import DistilBertModel, DistilBertConfig
import torch.nn as nn
import torch

class EncoderModel(nn.Module):
    def __init__(self):
        super(EncoderModel, self).__init__()
        self.config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
        self.encoder = DistilBertModel(self.config)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        linear_output = self.linear(x)
        logits = self.sigmoid(linear_output)
        return logits


class Summarizer(nn.Module):
    def __init__(self, encoder, classifier, device):
        super(Summarizer, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.device = device

    def extract_cls_embeddings(self, encoded_output, cls_idx):
        # Assume encoded_output shape: [batch_size, sequence_length, hidden_size]
        # cls_idx is a list of lists of indices for [CLS] tokens

        # Batch size and number of sentences (assuming each example has the same number of sentences)
        batch_size = encoded_output.shape[0]
        num_sentences = len(cls_idx[0]) if batch_size > 1 else len(cls_idx)

        # Initialize an empty tensor to store the [CLS] embeddings
        cls_embeddings = torch.zeros(batch_size, num_sentences, encoded_output.shape[2])

        # Handle each example in the batch separately
        for batch_idx in range(batch_size):
            for i, idx in enumerate(cls_idx[batch_idx]):
                cls_embeddings[batch_idx, i, :] = encoded_output[batch_idx, idx, :]

        return cls_embeddings



    def forward(self, input_ids, attention_mask, cls_idx):
        encoded_output = self.encoder(input_ids, attention_mask)
        encoded_output = self.extract_cls_embeddings(encoded_output, cls_idx)
        logits = self.classifier(encoded_output)
        return logits
