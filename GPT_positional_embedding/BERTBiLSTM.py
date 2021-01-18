from transformers import BertModel, AdamW, BertConfig
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

class BERTBiLSTM(nn.Module):
    def __init__(self, tokenizer, max_length=512, num_labels=42, embedding_dim=20, num_recurrent_layers=1,
                 bidirectional=True, lstm_hidden_size=768, mlp_hidden_size=300) -> None:
        # super(BERT_BiLSTM, self).__init__()
        # Feed at [CLS] token

        super().__init__()
        self.tokenizer = tokenizer
        self.config = BertConfig.from_pretrained('bert-base-cased')
        self.config.output_hidden_states = True
        self.config.output_attentions = True

        self.bert = BertModel.from_pretrained("bert-base-cased",
                                              config=self.config
                                              # , add_pooling_layer=False
                                              )
        self.bert.resize_token_embeddings(len(tokenizer))
        self.dropout = nn.Dropout(0.1)
        self.num_labels = num_labels
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.Embedding(num_embeddings=max_length * 2,
                                            embedding_dim=embedding_dim,
                                            padding_idx=0
                                            )

        self.num_recurrent_layers = num_recurrent_layers
        self.bidirectional = bidirectional
        self.lstm_hidden_size = lstm_hidden_size

        self.lstm = nn.LSTM(self.lstm_hidden_size + self.embedding_dim + self.embedding_dim,
                            hidden_size=self.lstm_hidden_size,
                            batch_first=True,
                            bidirectional=True
                            )

        self.mlp = nn.Linear(in_features=self.lstm_hidden_size * 2, out_features=mlp_hidden_size)
        self.classifier = nn.Linear(in_features=mlp_hidden_size, out_features=self.num_labels)

    def forward(self, device, input_ids, attention_mask, positional_ids, seq_lengths, sequence_length, labels=None,
                token_type_ids=None, tags=None):
        # outputs: (last_encoder_layer, pooled_output, attention_weight)
        # Set add_pooling_layer
        bert_outputs = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 encoder_hidden_states=torch.FloatTensor(
                                     (input_ids.shape[0], sequence_length, sequence_length))
                                 )

        sequence_output = bert_outputs[0]  # (B, P, H)
        # hidden_output = bert_outputs[2]

        for i in range(len(input_ids)):
            sequence_output[i, seq_lengths[i]:] = 0

        # LSTM
        ps = positional_ids[:, 0, :]
        po = positional_ids[:, 1, :]

        ps = self.embedding_layer(ps)
        po = self.embedding_layer(po)

        lstm_input = torch.cat((sequence_output, ps, po), dim=-1)  # (B, P, H+E+E)

        packed_lstm_input = pack_padded_sequence(lstm_input, seq_lengths.tolist(), batch_first=True,
                                                 enforce_sorted=False)

        h0 = torch.zeros(self.num_recurrent_layers * 2, input_ids.shape[0], self.lstm_hidden_size).to(device)
        c0 = torch.zeros(self.num_recurrent_layers * 2, input_ids.shape[0], self.lstm_hidden_size).to(device)

        packed_lstm_output, (lstm_hidden, lstm_cell) = self.lstm(packed_lstm_input, (h0, c0))  # (B, P, H*2) ...

        # mlp
        mlp_input = torch.cat((lstm_hidden[-2, :, :], lstm_hidden[-1, :, :]), dim=1)
        mlp_output = self.mlp(mlp_input)  # (B, H*2)

        # last layer
        # calcuate logits
        logits = self.classifier(mlp_output)

        # calcuate loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return logits, loss
