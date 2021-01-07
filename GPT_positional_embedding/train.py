import tokenizer
from GPTDecoder import GPTDecoder
from data import get_data_loader
from BERTBiLSTM import BERTBiLSTM


def train(train_sentences, train_masked_sentences, train_labels,
          dev_sentences, dev_masked_sentences, dev_labels,
          pretrained_model_name, max_len):
    # fix max len and padding, should note that still did not implement endoftext
    train_gpt_input_ids, train_gpt_attention_masks, train_gpt_positional_ids = tokenizer.get_gpt_token_ids(train_sentences, pretrained_model_name, max_len)

    decoder = GPTDecoder(pretrained_model_name, max_len)
    train_token_probabilities = decoder(train_gpt_input_ids)

    # token_ids to token?

    bert_tokenizer = tokenizer.get_bert_tokenizer()
    get_data_loader(bert_tokenizer, train_masked_sentences, )
