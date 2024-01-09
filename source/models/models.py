from transformers import DistilBertModel, DistilBertConfig
import torch.nn as nn
import torch
import numpy as np
from models.our_transformers import TransformerEncoder

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

class TransformerClassifier(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout):
        super(TransformerClassifier, self).__init__()
        self.transformer = TransformerEncoder(d_model, num_heads, d_ff, dropout, num_layers)
        
    def forward(self, x, mask):
        return self.transformer(x, mask)


class Summarizer(nn.Module):
    def __init__(self, encoder, classifier, device, classifier_type='linear'):
        super(Summarizer, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.device = device
        self.classifier_type = classifier_type

    def extract_cls_embeddings(self, encoded_output, cls_idx):
        # Assume encoded_output shape: [batch_size, sequence_length, hidden_size]
        # cls_idx is a list of lists of indices for [CLS] tokens

        # Batch size and number of sentences (as the max number of sentences in all batch)
        batch_size = encoded_output.shape[0]
        num_sentences = np.max([len(x) for x in cls_idx])

        # Initialize an empty tensor to store the [CLS] embeddings
        cls_embeddings = torch.zeros(batch_size, num_sentences, encoded_output.shape[2], device=self.device)

        # Handle each example in the batch separately
        for batch_idx in range(batch_size):
            for i, idx in enumerate(cls_idx[batch_idx]):
                cls_embeddings[batch_idx, i, :] = encoded_output[batch_idx, idx, :]

        return cls_embeddings

    def forward(self, input_ids, attention_mask, cls_idx):
        encoded_output = self.encoder(input_ids, attention_mask)
        encoded_output = encoded_output * cls_idx.unsqueeze(2).float()

        if self.classifier_type == 'linear':
            logits = self.classifier(encoded_output)
        elif self.classifier_type == 'transformer':
            logits = self.classifier(encoded_output, cls_idx)
        return logits