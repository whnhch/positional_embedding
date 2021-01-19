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

    def get_token_probs(self, input_ids, probs):
        output_prob = []
        # probs only of sequences
        # batch가 없다면
        for b in range(len(input_ids)):
            seq_prob = []
            for i in range(self.maxlen):
                # consider mask!!!
                tok_prob = probs[b, i, input_ids[b, :]]  # (1, H)
                seq_prob.append(tok_prob)  # (i, H)
            output_prob.append(seq_prob)  # ( b, P', P')

        return torch.tensor(output_prob)

    def __call__(self, input_ids):  # (B, P)
        last_hidden_state = self.get_decoder_output(input_ids)  # (B, P', H)
        logits = self.get_logits(last_hidden_state)  # (B, P', len(vocab))
        probs = self.get_probs(logits)  # (B, P', len(vocab))

        output_prob = self.get_token_probs(input_ids, probs)  # (B, P', P')

        return output_prob

    def get_words_probs(self, tokenizer, input_ids, probs):
        # without batch

        # convert id to token
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        words = []
        words_probs = []
        # convert token to word
        i = 0
        while i < len(tokens):
            if '#' in tokens[i]:
                wrecked_tokens = [tokens[i]]
                wrecked_probs = [probs[i]]
                for j in range(i + 1, len(tokens)):
                    if '#' in tokens[j]:
                        wrecked_tokens.append(tokens[j])
                        wrecked_probs.append(probs[j])
                    else:
                        wrecked_tokens.append(tokens[j])
                        wrecked_probs.append(probs[j])

                        word = tokenizer.convert_tokens_to_string(wrecked_tokens)
                        words.append(word)

                        # Use probability by last word
                        words_probs.append(wrecked_tokens[-1])

                        # Average
                        # words_probs.append(sum(wrecked_tokens)/len(wrecked_tokens))
                        i = j + 1
                        break
            else:
                words.append(tokens[i])
                words_probs.append(probs[i])
                i += 1

        return words, words_probs
