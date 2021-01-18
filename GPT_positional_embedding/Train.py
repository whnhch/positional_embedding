import Tokenizer
from GPTDecoder import GPTDecoder
from Data import get_data_loader
from BERTBiLSTM import BERTBiLSTM
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import time
import datetime

EPOCHS = 3
LEARNING_RATE = 1e-5
EPSILON = 1e-12


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train(device, train_sentences, train_masked_sentences, train_labels,
          dev_sentences, dev_masked_sentences, dev_labels,
          pretrained_model_name, max_len):
    # fix max len and padding, should note that still did not implement endoftext
    gpt_tokenizer = Tokenizer.get_gpt_tokenizer(pretrained_model_name)
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
    train_gpt_input_ids = Tokenizer.get_gpt_token_ids(
        gpt_tokenizer, train_sentences, max_len)

    decoder = GPTDecoder(pretrained_model_name, max_len)
    tr_token_probabilities = decoder(train_gpt_input_ids)
    tr_words, tr_words_probabilities = decoder.get_words_probs(gpt_tokenizer, train_gpt_input_ids, tr_token_probabilities)
    print(tr_words_probabilities)
    # tr_token_probabilities 는 gpt의 tokenizer에 따라 tokenizing된 상태이다. 이를 바꾸든지 아니면 같은 tokenizer를 사용해야 한다.
    # token_ids to token?

    bert_tokenizer = Tokenizer.get_bert_tokenizer
    train_dataloader = get_data_loader(bert_tokenizer, train_masked_sentences, train_labels, tr_words, tr_token_probabilities,
                                       max_len, data_type='train', tokenizer_type='BERT')

    model = BERTBiLSTM(bert_tokenizer, max_length=450)
    model.to(device)

    optimizer = AdamW(model.parameters(),
                      lr=LEARNING_RATE,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=EPSILON  # args.adam_epsilon  - default is 1e-8.
                      )

    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    training_stats = []

    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(EPOCHS):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHS))
        print('Training...')

        t0 = time.time()

        total_train_loss = []
        total_train_accuracy = []

        model.train()

        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_input_positional_ids = batch[2].to(device)
            b_input_seq_lengths = batch[3].to(device)
            b_labels = batch[4].to(device)

            b_input_seq_lengths, indicies = torch.sort(b_input_seq_lengths, dim=0, descending=True)

            def smart_sort(x, per):
                z = torch.empty_like(x)
                for i in range(len(per)):
                    z[per[i]] = x[i]
                return z

            b_input_ids = smart_sort(b_input_ids, indicies)
            b_input_mask = smart_sort(b_input_mask, indicies)
            b_input_positional_ids = smart_sort(b_input_positional_ids, indicies)
            b_labels = smart_sort(b_labels, indicies)

            logits, loss = model(
                b_input_ids,
                attention_mask=b_input_mask,
                positional_ids=b_input_positional_ids,
                seq_lengths=b_input_seq_lengths,
                sequence_length=max_len,
                labels=b_labels,
                token_type_ids=None
            )

            logits = F.softmax(logits, dim=1)

            total_train_loss.append(loss.item())

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

            logits_ = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_train_accuracy.append(flat_accuracy(logits_, label_ids))

            if step % 128 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)

                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:},     Loss: {:}.      Accuracy: {:}'.format(step, len(
                    train_dataloader), elapsed, sum(total_train_loss[:step]) / step, sum(
                    total_train_accuracy[:step]) / step))

        avg_train_accuracy = sum(total_train_accuracy) / len(train_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_train_accuracy))

        avg_train_loss = sum(total_train_loss) / len(train_dataloader)

        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        SAVE_EPOCH = epoch_i
        SAVE_PATH = "model_gpt_bertbilstm_" + str(epoch_i) + ".pt"
        SAVE_LOSS = loss

        torch.save({
            'epoch': SAVE_EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': SAVE_LOSS,
        }, SAVE_PATH)

        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        predictions_, true_labels_ = [], []

        # Evaluate data for one epoch
        for batch in dev_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_input_positional_ids = batch[2].to(device)
            b_input_seq_lengths = batch[3].to(device)
            b_labels = batch[4].to(device)

            b_input_seq_lengths, indicies = torch.sort(b_input_seq_lengths, dim=0, descending=True)

            b_input_ids = smart_sort(b_input_ids, indicies)
            b_input_mask = smart_sort(b_input_mask, indicies)
            b_input_positional_ids = smart_sort(b_input_positional_ids, indicies)
            b_labels = smart_sort(b_labels, indicies)

            with torch.no_grad():
                logits, loss = model(b_input_ids,
                                     attention_mask=b_input_mask,
                                     positional_ids=b_input_positional_ids,
                                     seq_lengths=b_input_seq_lengths,
                                     sequence_length=max_len,
                                     labels=b_labels,
                                     token_type_ids=None
                                     )
                logits = F.softmax(logits, dim=1)

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            predictions_.append(np.argmax(logits, axis=1))
            true_labels_.append(label_ids)

            total_eval_accuracy += flat_accuracy(logits, label_ids)

        if (epoch_i == 2):
            y_pred, y_true = [], []
            row = len(predictions_)
            for i in range(row):
                col = len(predictions_[i])
                for j in range(col):
                    tmp1 = predictions_[i][j]
                    tmp2 = true_labels_[i][j]
                    y_pred.append(label_list[int(tmp1)])
                    y_true.append(label_list[int(tmp2)])

            df_result = pd.DataFrame(y_pred)
            df_result.to_csv('./dataset/dev_pred.csv', header=None, index=False)

            df_result = pd.DataFrame(y_true)
            df_result.to_csv('./dataset/dev_gt.csv', header=None, index=False)

        # Report the final accuracy for this validation run.
        avg_dev_accuracy = total_eval_accuracy / len(dev_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_dev_accuracy))

        # Calculate the average loss over all of the batches.
        avg_dev_loss = total_eval_loss / len(dev_dataloader)

        # Measure how long the validation run took.
        dev_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_dev_loss))
        print("  Validation took: {:}".format(dev_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_dev_loss,
                'Valid. Accur.': avg_dev_accuracy,
                'Training Time': training_time,
                'Validation Time': dev_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
