from transformers import DistilBertModel, DistilBertConfig
import torch.nn as nn
import torch
import numpy as np
from models.our_transformers import TransformerEncoder
import nltk
nltk.download('punkt')
from nltk import sent_tokenize
from transformers import DistilBertTokenizer
from preprocessing.preprocess import split_into_chunks

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
    def __init__(self, device, classifier_type='linear'):
        super(Summarizer, self).__init__()
        self.encoder = EncoderModel()
        if classifier_type == 'linear':
            self.classifier = LinearClassifier(768, 1)
        elif classifier_type == 'transformer':
            self.classifier = TransformerClassifier(768, 1, 1, 768, 0.1)
        self.device = device
        self.classifier_type = classifier_type
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

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
        input_ids = input_ids.long()
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
    
    def preprocess(self, text, max_length=512):
        sentences = sent_tokenize(text)
        tokenized_text = ['[CLS] ' + sent + ' [SEP]' for sent in sentences]
        full_text = ' '.join(tokenized_text)
        encoded_dict = self.tokenizer.encode_plus(
            full_text,
            add_special_tokens=False,  # Since [CLS] and [DEL] are already added in sentences
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
            max_length=max_length
        )
        input_ids = encoded_dict['input_ids'][0]
        attention_masks = encoded_dict['attention_mask'][0]
        cls_idx = torch.zeros(len(input_ids), dtype=torch.bool)
        for i, token_id in enumerate(input_ids):
            if token_id == self.tokenizer.cls_token_id:
                cls_idx[i] = True
        return input_ids, attention_masks, cls_idx
    
    def summarize(self, sum_len, text, isTaken):
        # process each chunk
        input_ids, attention_masks, cls_idx = self.preprocess(text)
        # print('n_tokens', len(input_ids))
        input_ids = input_ids.unsqueeze(0)
        attention_masks = attention_masks.unsqueeze(0)
        cls_idx = cls_idx.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)
        cls_idx = cls_idx.to(self.device)
        logits = self.forward(input_ids, attention_masks, cls_idx)
        logits = logits.squeeze(0)
        logits = logits.cpu().detach().numpy()
        summary = np.argsort(logits, axis=0)[-sum_len:]
        top_logits_values = [float(i) for i in logits[summary]]
        summary = summary.tolist()
        summary = [sent_tokenize(text)[i] if isinstance(i, int) else sent_tokenize(text)[i[0]] for i in summary]
        if not isTaken:
            summary = ' '.join(summary)
        return summary, logits, top_logits_values

    def predict(self, text, sum_len=6, method='sum_all', n_sent=1, ch_sum_len = 1):
        """
        ch_sum_len - number of sentences to output for summaries of chunks
        sum_len - number of sentences to output for the summary
        method, can be either sum_all (summarize all summaries) or
            takeN - take N highest-score sentences, using n_sent
        """
        # split the text into chunks
        chunks = split_into_chunks(self.tokenizer, text)
        summaries = []
        logitses = []
        # print('chunks', chunks)
        # print('len chunks', len(chunks))

        if len(chunks) == 1:
            print('chunk 1')
            # initial text is under 512 tokens
            ch_text = " ".join(chunks[0])
            summary, logits, _ = self.summarize(sum_len, ch_text, False)

        else:
            for j, chunk in enumerate(chunks):
                ch_text = " ".join(chunk)
                # print('chunk', len(ch_text), j)
                summary, logits, top_logits_values = self.summarize(ch_sum_len, ch_text, True)
                summaries.append(summary)
                logitses.append(top_logits_values)
            
            if method == 'sum_all':
                new_summaries = []
                for summary in summaries:
                    new_summaries.extend(summary)
                sum_text = " ".join(new_summaries)
                summary, logits, _ = self.summarize(sum_len, sum_text, False)

            elif method == 'takeN':
                summaries = [summary[-n_sent:] for summary in summaries]
                summaries = [item for sublist in summaries for item in sublist]
                logitses = [logits[-n_sent:] for logits in logitses]
                logitses = [item for sublist in logitses for item in sublist]
                logit_indexes = sorted(range(len(logitses)), key=lambda i: logitses[i], reverse=True)[:sum_len]
                summaries = [summaries[i] for i in logit_indexes]
                # print('len_sum', len(summaries))
                logitses = [logitses[i] for i in logit_indexes]
                summary = " ".join(summaries)
                logits = logitses

        return summary, logits

