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
        masked_output = encoded_output * cls_idx.unsqueeze(-1).float()

        # Determine the maximum number of [CLS] tokens in the batch
        max_num_cls = max(cls_idx.sum(dim=1))

        # Initialize a list to hold the padded [CLS] embeddings for each batch
        padded_cls_embeddings = []

        for i in range(masked_output.size(0)):
            cls_embeddings = masked_output[i]
            non_zero_embeddings = cls_embeddings[cls_embeddings.sum(dim=1) != 0]

            # Pad the non-zero embeddings to the same length
            num_padding = max_num_cls - non_zero_embeddings.size(0)
            padding = torch.zeros(num_padding, non_zero_embeddings.size(1), device=non_zero_embeddings.device)
            padded_embeddings = torch.cat([non_zero_embeddings, padding], dim=0)
            padded_cls_embeddings.append(padded_embeddings)

        # Stack the padded embeddings to get a tensor of shape [batch_size, max_num_cls, 768]
        encoded_output = torch.stack(padded_cls_embeddings, dim=0)
        return encoded_output

    def forward(self, input_ids, attention_mask, cls_idx):
        # input_ids shape: [batch_size, num_tokens]
        # attention_mask shape: [batch_size, num_tokens]
        # cls_idx shape: [batch_size, num_tokens]
        encoded_output = self.encoder(input_ids, attention_mask)
        # encoded_output shape: [batch_size, num_tokens, 768]
        
        # Extract the [CLS] embeddings
        encoded_output = self.extract_cls_embeddings(encoded_output, cls_idx)
        # encoded_output shape: [batch_size, max_num_sentences, 768]

        if self.classifier_type == 'linear':
            logits = self.classifier(encoded_output)
        elif self.classifier_type == 'transformer':
            logits = self.classifier(encoded_output, cls_idx)
        return logits