import numpy as np
import torch
import copy
from torch.nn import functional as F
from transformers import AutoTokenizer, RobertaForMaskedLM


class HookCloser:
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper

    def __call__(self, module, input_, output_):
        self.model_wrapper.curr_embedding = output_
        output_.retain_grad()

class RobertaClassifier(object):
    def __init__(self,  device, trigger_len, alpha, max_length=128, batch_size=8):
        self.alpha = alpha
        self.trigger_len = trigger_len
        self.model = RobertaForMaskedLM.from_pretrained('FacebookAI/roberta-large')
        self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")
        self.tokenizer.add_tokens(["<trigger>"])

        self.embedding_layer = self.model.roberta.embeddings.word_embeddings
        self.curr_embedding = None
        self.hook = self.embedding_layer.register_forward_hook(HookCloser(self))
        self.embedding = self.embedding_layer.weight.detach().cpu().numpy()

        self.word2id = dict()
        for i in range(self.tokenizer.vocab_size):
            self.word2id[self.tokenizer.convert_ids_to_tokens(i)] = i

        self.trigger = []
        self.max_length = max_length
        self.device = device
        self.model.to(device)
        self.batch_size = batch_size


    def set_trigger(self, trigger):
        self.trigger = trigger


    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)
    

    def get_prob(self, input_):
        return self.get_grad([self.tokenizer.tokenize(sent) for sent in input_], [0] * len(input_))[0]


    def get_grad(self, tokenized_input, labels):
        v = self.predict(tokenized_input, labels)
        return v[0], v[1]


    def predict(self, input_, labels=None, return_loss=False):
        sen_list = []      
        sen_list_tokens = [[] for _ in range(self.trigger_len)]
        mask_pos = []
        mask_pos_tokens = [[] for _ in range(self.trigger_len)]
        trigger_offset = []


        for text in input_:
            text = [tok.strip() for tok in text]
            trigger_index = text.index("<trigger>") 

            text_t = text[:trigger_index] + self.trigger + text[trigger_index + 1:]
            sen_list.append(text_t)
            mask_pos_tmp = text_t.index("<mask>")
            mask_pos.append(mask_pos_tmp + 1)
            for i in range(self.trigger_len):
                if i == 0:
                    text_t_token = text[:trigger_index] + ["<mask>"]
                else:
                    text_t_token = text[:trigger_index] + self.trigger[:i] + ["<mask>"]
                sen_list_tokens[i].append(text_t_token)
                mask_pos_tmp_token = text_t_token.index("<mask>")
                mask_pos_tokens[i].append(mask_pos_tmp_token + 1)

            trigger_offset.append(trigger_index + 1)

        sent_lens = [len(sen) for sen in sen_list]
        batch_len = max(sent_lens) + 2 
        sent_lens_tokens = [[len(sen) for sen in token_sen_list] for token_sen_list in sen_list_tokens]
        batch_len_tokens = [max(token_sent_lens) + 2 for token_sent_lens in sent_lens_tokens]

        attentions = np.array([[1] * (len(sen) + 2) + [0] * (batch_len - 2 - len(sen))
                            for sen in sen_list], dtype='int64')
        attentions_tokens = []
        for i, token_sen_list in enumerate(sen_list_tokens):
            batch_len_token = batch_len_tokens[i]
            if i == 0:
                attentions_token = np.array([[1] + [0] * (len(sen) - 1) + [1] * 2 + [0] * (batch_len_token - 2 - len(sen))
                                        for sen in token_sen_list], dtype='int64')
            else:
                attentions_token = np.array([[1] + [0] * (len(sen) - i) + [1] * (i + 1) + [0] * (batch_len_token - 2 - len(sen))
                                        for sen in token_sen_list], dtype='int64')
            attentions_tokens.append(attentions_token)

        sen_list = [self.tokenizer.convert_tokens_to_ids(sen) for sen in sen_list]
        sen_list_tokens = [[self.tokenizer.convert_tokens_to_ids(sen) for sen in token_sen_list] for token_sen_list in sen_list_tokens]

        input_ids = np.array([
            [self.tokenizer.cls_token_id] + sen + [self.tokenizer.sep_token_id] 
            + [self.tokenizer.pad_token_id] * (batch_len - 2 - len(sen)) for sen in sen_list], dtype='int64')
        input_ids_tokens = [np.array([
            [self.tokenizer.cls_token_id] + sen + [self.tokenizer.sep_token_id] 
            + [self.tokenizer.pad_token_id] * (batch_len_tokens[i] - 2 - len(sen)) for sen in token_sen_list], dtype='int64') for i, token_sen_list in enumerate(sen_list_tokens)]

        result = []
        result_grad = []
        if labels is None:
            labels = [0] * len(sen_list)
        labels = torch.LongTensor(labels).to(self.device)
        overall_loss = 0

        for i in range((len(sen_list) + self.batch_size - 1) // self.batch_size):
            curr_input_ids = input_ids[i * self.batch_size: (i + 1) * self.batch_size]
            curr_input_ids_tokens = [input_ids_token[i * self.batch_size: (i + 1) * self.batch_size] for input_ids_token in input_ids_tokens]

            curr_mask = attentions[i * self.batch_size: (i + 1) * self.batch_size]
            curr_mask_tokens = [attentions_token[i * self.batch_size: (i + 1) * self.batch_size] for attentions_token in attentions_tokens]

            curr_label = labels[i * self.batch_size: (i + 1) * self.batch_size]
            curr_label_tokens = []
            for token_id in range(len(self.trigger)):
                curr_label_token = copy.deepcopy(curr_label)
                curr_label_token[:] = self.tokenizer.convert_tokens_to_ids(self.trigger[token_id])
                curr_label_tokens.append(curr_label_token)

            curr_mask_pos = mask_pos[i * self.batch_size: (i + 1) * self.batch_size]
            curr_mask_pos_tokens = [mask_pos_token[i * self.batch_size: (i + 1) * self.batch_size] for mask_pos_token in mask_pos_tokens]

            curr_trigger_offset = trigger_offset[i * self.batch_size: (i + 1) * self.batch_size]

            xs = torch.from_numpy(curr_input_ids).long().to(self.device)
            xs_tokens = [torch.from_numpy(curr_input_ids_token).long().to(self.device) for curr_input_ids_token in curr_input_ids_tokens]

            masks = torch.from_numpy(curr_mask).long().to(self.device)
            masks_tokens = [torch.from_numpy(curr_mask_token).long().to(self.device) for curr_mask_token in curr_mask_tokens]
 
            losses = []
            grads_tmp_tokens = []
            # probs = []
            for j in range(self.trigger_len):
                output = self.model.roberta(input_ids=xs_tokens[j], attention_mask=masks_tokens[j], output_hidden_states=True)
                hidden = [output[0][idx, item, :] for idx, item in enumerate(curr_mask_pos_tokens[j])]
                hidden = torch.stack(hidden, dim=0)
                emb = self.model.lm_head.layer_norm(self.model.lm_head.dense(hidden))
                logits = self.model.lm_head.decoder(emb)
                prob = torch.softmax(logits, dim=1).detach().cpu()
                loss_token = self.alpha * torch.sum(F.cross_entropy(logits, curr_label_tokens[j], reduction="none"))
                if j == 0:
                    loss_token = loss_token.detach().cpu().numpy()
                    losses.append(loss_token)
                    self.curr_embedding = None
                    continue
                else:
                    loss_token.backward()
                    loss_token = loss_token.detach().cpu().numpy()
                    
                    grads_tmp_token = self.curr_embedding.grad.clone().cpu().numpy()
                    grads_tmp_token = [item[curr_trigger_offset[idx]:curr_trigger_offset[idx] +  j]
                                    for idx, item in enumerate(grads_tmp_token)]
                    grads_token = []
                    pad_width = ((0, len(self.trigger)-j), (0, 0))
                    for arr in grads_tmp_token:
                        padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
                        grads_token.append(padded_arr)
                    grads_tmp_tokens.append(grads_token)
                    losses.append(loss_token)
                    self.curr_embedding.grad.zero_()
                    self.curr_embedding = None

            outputs = self.model.roberta(input_ids=xs, attention_mask=masks, output_hidden_states=True)
            hidden = [outputs[0][idx, item, :] for idx, item in enumerate(curr_mask_pos)]
            hidden = torch.stack(hidden, dim=0)
            emb = self.model.lm_head.layer_norm(self.model.lm_head.dense(hidden))
            logits = self.model.lm_head.decoder(emb)
            prob = torch.softmax(logits, dim=1).detach().cpu()
            result.append(prob)

            loss_1 = F.cross_entropy(logits, curr_label, reduction="none")
            loss_1 = -loss_1.sum()
            loss_1.backward()
            loss_2 = np.sum(losses)
            loss = loss_1 + loss_2
            overall_loss += loss.detach().cpu().numpy()

            grads_tmp_1 = self.curr_embedding.grad.clone().cpu().numpy()
            grads_tmp_1 = [item[curr_trigger_offset[idx]:curr_trigger_offset[idx] + len(self.trigger)]
                        for idx, item in enumerate(grads_tmp_1)]
            grads_tmp_tokens.append(grads_tmp_1)
            grads_tmp = list(np.sum(grads_tmp_tokens,axis=0))
            result_grad.append(grads_tmp)
            
            self.curr_embedding.grad.zero_()
            self.curr_embedding = None
            
        result = np.concatenate(result, axis=0)
        result_grad = np.concatenate(result_grad, axis=0)
        if return_loss:
            return result, result_grad, overall_loss
        else:
            return result, result_grad


