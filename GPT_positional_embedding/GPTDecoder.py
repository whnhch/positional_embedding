import transformers
import torch
from transformers import GPT2Model
from PIL import Image


class GPTDecoder():
    def __init__(self, pretrained_model_name, max_len):
        self.decoder = GPT2Model.from_pretrained(pretrained_model_name)
        self.wte = self.decoder.get_input_embeddings().weight
        self.softmax = torch.nn.Softmax(dim=2)
        self.maxlen = max_len

    def get_decoder_output(self, inputs):
        outputs = self.decoder(inputs)
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state

    def get_logits(self, last_hidden_state):
        return torch.matmul(last_hidden_state, self.wte.T)  # (B,P,768)

    def get_probs(self, logits):
        return self.softmax(logits)

    def __call__(self, input_ids):  # (B, P)
        last_hidden_state = self.get_decoder_output(input_ids)  # (B, P', H)
        logits = self.get_logits(last_hidden_state)  # (B, P', len(vocab))
        probs = self.get_probs(logits)  # (B, P', len(vocab))

        output_prob = []
        # probs only of sequences
        for b in range(len(input_ids)):
            seq_prob = []
            for i in range(self.maxlen):
                # consider mask!!!
                tok_prob = probs[b, i, input_ids[b, :]]
                seq_prob.append(tok_prob)
            output_prob.append(seq_prob)

        # for i in range(self.maxlen):
        #     # consider mask!!!
        #     tok_prob = probs[:, i, input_ids[:,:]]  # (1, B, MAXLEN)
        #     output_prob.append(tok_prob)  # (i, B, 1, MAXLEN)
        # # output_prob # (MAXLEN, B, 1, MAXLEN)
        # # want # ( B, MAXLEN, MAXLEN)
        # output_prob.reshape((len(input_ids), self.maxlen, self.maxlen))


