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
## 在模型的前向传播过程中获取中间层的输出，model_wrapper用于存储模型的当前嵌入（embedding）输出，并且可以在需要时通过访问model_wrapper.curr_embedding来获取这些输出

class RobertaClassifier(object):
    def __init__(self, device, alpha, loss_type, max_length=128, batch_size=8):
        self.loss_type = loss_type
        self.alpha = alpha
        self.model = RobertaForMaskedLM.from_pretrained('/data/xuyue/advprompt/roberta/')
        self.tokenizer = AutoTokenizer.from_pretrained("/data/xuyue/advprompt/roberta/")
        self.tokenizer.add_tokens(["<trigger>"])

        self.embedding_layer = self.model.roberta.embeddings.word_embeddings#将roberta的embedding层作为self的embedding_layer
        self.curr_embedding = None
        self.hook = self.embedding_layer.register_forward_hook(HookCloser(self)) #储存前向传播过程中的中间层输出
        self.embedding = self.embedding_layer.weight.detach().cpu().numpy()

        self.word2id = dict()
        for i in range(self.tokenizer.vocab_size):
            self.word2id[self.tokenizer.convert_ids_to_tokens(i)] = i #使用AutoTokenizer，建立word2id的字典，方便对应

        self.trigger = []
        self.max_length = max_length
        self.device = device
        self.model.to(device)
        self.batch_size = batch_size

    def set_trigger(self, trigger):
        """Set the trigger.

        :param trigger: a list of tokens as the trigger. Each token must be a word in the vocabulary.
        """
        self.trigger = trigger

    def get_pred(self, input_):
        """Get prediction of the masked position in each sentence.

        :param input_: a list of sentences, each contains one <mask> token and one <trigger> token.
        :return: a list of integers meaning the predicted token id for the masked position.
        """
        return self.get_prob(input_).argmax(axis=1) #axis=1,返回列index，对应概率最大的mask word.

    def get_prob(self, input_):
        """Get prediction of the probability of words for the masked position in each sentence.

        :param input_: a list of sentences, each contains one <mask> token and one <trigger> token.
        :return: a matrix of n * v, where n is the number of sentences, and v is the number of words.
        """
        return self.get_grad([self.tokenizer.tokenize(sent) for sent in input_], [0] * len(input_))[0]

    def get_grad(self, tokenized_input, labels):
        """Get prediction of the probability of words for the masked position in each sentence, and the gradient of
        loss with respect to the trigger tokens.

        :param tokenized_input: a list of tokenized sentences. Each contains one <mask> token and one <trigger> token.
        :param labels: a list of integers showing the target word id.
        :return: A tuple of two matrices. The first n * v matrix shows the predicted probability distribution on the
            masked position. The second n * l * d matrix shows the gradient of the specified loss with respect to the
            word embedding of the trigger tokens.
        """
        v = self.predict(tokenized_input, labels)
        return v[0], v[1]

    def predict(self, input_, labels=None, return_loss=False):
        """Implementation of get_grad with optional loss in return."""
        sen_list = []          # a copy of input with <trigger> being replaced by actual trigger tokens
        sen_list_token0 = []
        sen_list_token1 = []
        sen_list_token2 = []
        sen_list_token3 = []
        sen_list_token4 = []


        mask_pos = []          # the index of <mask> in each sentence
        mask_pos_token0 = [] 
        mask_pos_token1 = [] 
        mask_pos_token2 = []
        mask_pos_token3 = [] 
        mask_pos_token4 = [] 


        trigger_offset = []    # the index of the first trigger token

        sen_list_no_trigger = []    # a copy of input with <trigger> removed
        mask_pos_no_trigger = []    # the index of <mask> in each sentence (it can be different from mask_pos.)
        
        for text in input_:#将"<trigger>"替换成真实的trigger值
            text = [tok.strip() for tok in text]
            trigger_index = text.index("<trigger>") 

            text_t = text[:trigger_index] + self.trigger + text[trigger_index + 1:]
            text_t_token0 = text[:trigger_index] + ["<mask>"]
            text_t_token1 = text[:trigger_index] + self.trigger[0:1] + ["<mask>"]
            text_t_token2 = text[:trigger_index] + self.trigger[0:2]+ ["<mask>"]
            text_t_token3 = text[:trigger_index] + self.trigger[0:3] + ["<mask>"]
            text_t_token4 = text[:trigger_index] + self.trigger[0:4]+ ["<mask>"]

            sen_list.append(text_t)
            sen_list_token0.append(text_t_token0)
            sen_list_token1.append(text_t_token1)
            sen_list_token2.append(text_t_token2)
            sen_list_token3.append(text_t_token3)
            sen_list_token4.append(text_t_token4)


            mask_pos_tmp = text_t.index("<mask>")
            mask_pos_tmp_token0 = text_t_token0.index("<mask>")
            mask_pos_tmp_token1 = text_t_token1.index("<mask>")
            mask_pos_tmp_token2 = text_t_token2.index("<mask>")
            mask_pos_tmp_token3 = text_t_token3.index("<mask>")
            mask_pos_tmp_token4 = text_t_token4.index("<mask>")



            # a CLS token will be added to the begining
            mask_pos.append(mask_pos_tmp + 1)
            mask_pos_token0.append(mask_pos_tmp_token0 + 1)
            mask_pos_token1.append(mask_pos_tmp_token1 + 1)
            mask_pos_token2.append(mask_pos_tmp_token2 + 1)
            mask_pos_token3.append(mask_pos_tmp_token3 + 1)
            mask_pos_token4.append(mask_pos_tmp_token4 + 1)



            trigger_offset.append(trigger_index + 1)
            text_t_no_trigger = text[:trigger_index] + text[trigger_index + 1:]
            sen_list_no_trigger.append(text_t_no_trigger)
            mask_pos_no_trigger.append(text_t_no_trigger.index("<mask>") + 1)

        sent_lens = [len(sen) for sen in sen_list]
        batch_len = max(sent_lens) + 2 #list中最长的句子长度（加上cls和sep）

        sent_lens_token0 = [len(sen) for sen in sen_list_token0]
        batch_len_token0 = max(sent_lens_token0) + 2
        sent_lens_token1 = [len(sen) for sen in sen_list_token1]
        batch_len_token1 = max(sent_lens_token1) + 2
        sent_lens_token2 = [len(sen) for sen in sen_list_token2]
        batch_len_token2 = max(sent_lens_token2) + 2
        sent_lens_token3 = [len(sen) for sen in sen_list_token3]
        batch_len_token3 = max(sent_lens_token3) + 2
        sent_lens_token4 = [len(sen) for sen in sen_list_token4]
        batch_len_token4 = max(sent_lens_token4) + 2

        attentions = np.array([[1] * (len(sen) + 2) + [0] * (batch_len - 2 - len(sen))
                               for sen in sen_list], dtype='int64')
        attentions_token0 = np.array([[1] + [0] * (len(sen)-1) +[1] * 2 + [0] * (batch_len_token0 - 2 - len(sen))
                               for sen in sen_list_token0], dtype='int64')
        attentions_token1 = np.array([[1] + [0] * (len(sen)-1) +[1] * 2 + [0] * (batch_len_token1 - 2 - len(sen))
                               for sen in sen_list_token1], dtype='int64')
        attentions_token2 = np.array([[1] + [0] * (len(sen)-2) +[1] * 3 + [0] * (batch_len_token2 - 2 - len(sen))
                               for sen in sen_list_token2], dtype='int64')
        attentions_token3 = np.array([[1] + [0] * (len(sen)-3) +[1] * 4 + [0] * (batch_len_token3 - 2 - len(sen))
                               for sen in sen_list_token3], dtype='int64')
        attentions_token4 = np.array([[1] + [0] * (len(sen)-4) +[1] * 5 + [0] * (batch_len_token4 - 2 - len(sen))
                               for sen in sen_list_token4], dtype='int64')
        
        sen_list = [self.tokenizer.convert_tokens_to_ids(sen) for sen in sen_list]#将token转换成id

        sen_list_token0 = [self.tokenizer.convert_tokens_to_ids(sen) for sen in sen_list_token0]
        sen_list_token1 = [self.tokenizer.convert_tokens_to_ids(sen) for sen in sen_list_token1]
        sen_list_token2 = [self.tokenizer.convert_tokens_to_ids(sen) for sen in sen_list_token2]
        sen_list_token3 = [self.tokenizer.convert_tokens_to_ids(sen) for sen in sen_list_token3]
        sen_list_token4 = [self.tokenizer.convert_tokens_to_ids(sen) for sen in sen_list_token4]



        sen_list_no_trigger = [self.tokenizer.convert_tokens_to_ids(sen) for sen in sen_list_no_trigger]
        
        input_ids = np.array([
            [self.tokenizer.cls_token_id] + sen + [self.tokenizer.sep_token_id]
            + [self.tokenizer.pad_token_id] * (batch_len - 2 - len(sen)) for sen in sen_list], dtype='int64')
        
        input_ids_token0 = np.array([
            [self.tokenizer.cls_token_id] + sen + [self.tokenizer.sep_token_id]
            + [self.tokenizer.pad_token_id] * (batch_len_token0 - 2 - len(sen)) for sen in sen_list_token0], dtype='int64')
        
        input_ids_token1 = np.array([
            [self.tokenizer.cls_token_id] + sen + [self.tokenizer.sep_token_id]
            + [self.tokenizer.pad_token_id] * (batch_len_token1 - 2 - len(sen)) for sen in sen_list_token1], dtype='int64')
        
        input_ids_token2 = np.array([
            [self.tokenizer.cls_token_id] + sen + [self.tokenizer.sep_token_id]
            + [self.tokenizer.pad_token_id] * (batch_len_token2 - 2 - len(sen)) for sen in sen_list_token2], dtype='int64')
        
        input_ids_token3 = np.array([
            [self.tokenizer.cls_token_id] + sen + [self.tokenizer.sep_token_id]
            + [self.tokenizer.pad_token_id] * (batch_len_token3 - 2 - len(sen)) for sen in sen_list_token3], dtype='int64')
        
        input_ids_token4 = np.array([
            [self.tokenizer.cls_token_id] + sen + [self.tokenizer.sep_token_id]
            + [self.tokenizer.pad_token_id] * (batch_len_token4 - 2 - len(sen)) for sen in sen_list_token4], dtype='int64')
        
        #'''INPUT格式: [(cls + id of text1 + sep + pad) , (cls + id of text2 + sep + pad) ,...] '''
        input_ids_no_trigger = np.array([
            [self.tokenizer.cls_token_id] + sen + [self.tokenizer.sep_token_id]
            + [self.tokenizer.pad_token_id] * (batch_len - 2 - len(sen) - len(self.trigger))
            for sen in sen_list_no_trigger], dtype='int64')


        result = []
        result_grad = []

        if labels is None:
            labels = [0] * len(sen_list)
        labels = torch.LongTensor(labels).to(self.device)
        
        overall_loss = 0

        for i in range((len(sen_list) + self.batch_size - 1) // self.batch_size):
            curr_input_ids = input_ids[i * self.batch_size: (i + 1) * self.batch_size]
            curr_input_ids_token0 = input_ids_token0[i * self.batch_size: (i + 1) * self.batch_size]
            curr_input_ids_token1 = input_ids_token1[i * self.batch_size: (i + 1) * self.batch_size]
            curr_input_ids_token2 = input_ids_token2[i * self.batch_size: (i + 1) * self.batch_size]
            curr_input_ids_token3 = input_ids_token3[i * self.batch_size: (i + 1) * self.batch_size]
            curr_input_ids_token4 = input_ids_token4[i * self.batch_size: (i + 1) * self.batch_size]


            curr_mask = attentions[i * self.batch_size: (i + 1) * self.batch_size]
            curr_mask_token0 = attentions_token0[i * self.batch_size: (i + 1) * self.batch_size]
            curr_mask_token1 = attentions_token1[i * self.batch_size: (i + 1) * self.batch_size]
            curr_mask_token2 = attentions_token2[i * self.batch_size: (i + 1) * self.batch_size]
            curr_mask_token3 = attentions_token3[i * self.batch_size: (i + 1) * self.batch_size]
            curr_mask_token4 = attentions_token4[i * self.batch_size: (i + 1) * self.batch_size]


            curr_label = labels[i * self.batch_size: (i + 1) * self.batch_size]
            curr_label_token0 = copy.deepcopy(curr_label)
            curr_label_token1 = copy.deepcopy(curr_label)
            curr_label_token2 = copy.deepcopy(curr_label)
            curr_label_token3 = copy.deepcopy(curr_label)
            curr_label_token4 = copy.deepcopy(curr_label)
            curr_label_token0[:] = self.tokenizer.convert_tokens_to_ids(self.trigger[0])
            curr_label_token1[:] = self.tokenizer.convert_tokens_to_ids(self.trigger[1])
            curr_label_token2[:] = self.tokenizer.convert_tokens_to_ids(self.trigger[2])
            curr_label_token3[:] = self.tokenizer.convert_tokens_to_ids(self.trigger[3])
            curr_label_token4[:] = self.tokenizer.convert_tokens_to_ids(self.trigger[4])


            curr_mask_pos_no_trigger = mask_pos_no_trigger[i * self.batch_size: (i + 1) * self.batch_size]
            curr_mask_pos = mask_pos[i * self.batch_size: (i + 1) * self.batch_size]
            curr_mask_pos_token0 = mask_pos_token0[i * self.batch_size: (i + 1) * self.batch_size]
            curr_mask_pos_token1 = mask_pos_token1[i * self.batch_size: (i + 1) * self.batch_size]
            curr_mask_pos_token2 = mask_pos_token2[i * self.batch_size: (i + 1) * self.batch_size]
            curr_mask_pos_token3 = mask_pos_token3[i * self.batch_size: (i + 1) * self.batch_size]
            curr_mask_pos_token4 = mask_pos_token4[i * self.batch_size: (i + 1) * self.batch_size]


            curr_trigger_offset = trigger_offset[i * self.batch_size: (i + 1) * self.batch_size]
            curr_input_ids_no_trigger = input_ids_no_trigger[i * self.batch_size: (i + 1) * self.batch_size]

            # ===== compute output embed without trigger if loss is embdis.
            if self.loss_type == "embdis":
                xs = torch.from_numpy(curr_input_ids_no_trigger).long().to(self.device)
                masks = torch.from_numpy(curr_mask).long().to(self.device)
                outputs = self.model.roberta(input_ids=xs, attention_mask=masks[:, len(self.trigger):],
                                             output_hidden_states=True)
                hidden = [outputs[0][idx, item, :] for idx, item in enumerate(curr_mask_pos_no_trigger)]
                hidden = torch.stack(hidden, dim=0)
                emb_no_trigger = self.model.lm_head.layer_norm(self.model.lm_head.dense(hidden))
            else:
                emb_no_trigger = None
            # =======================================

            xs = torch.from_numpy(curr_input_ids).long().to(self.device)
            xs_token0 = torch.from_numpy(curr_input_ids_token0).long().to(self.device)
            xs_token1 = torch.from_numpy(curr_input_ids_token1).long().to(self.device)
            xs_token2 = torch.from_numpy(curr_input_ids_token2).long().to(self.device)
            xs_token3 = torch.from_numpy(curr_input_ids_token3).long().to(self.device)
            xs_token4 = torch.from_numpy(curr_input_ids_token4).long().to(self.device)


            masks = torch.from_numpy(curr_mask).long().to(self.device)
            masks_token0 = torch.from_numpy(curr_mask_token0).long().to(self.device)
            masks_token1 = torch.from_numpy(curr_mask_token1).long().to(self.device)
            masks_token2 = torch.from_numpy(curr_mask_token2).long().to(self.device)
            masks_token3 = torch.from_numpy(curr_mask_token3).long().to(self.device)
            masks_token4 = torch.from_numpy(curr_mask_token4).long().to(self.device)

            
            outputs_token0 = self.model.roberta(input_ids=xs_token0, attention_mask=masks_token0, output_hidden_states=True)
            hidden_token0 = [outputs_token0[0][idx, item, :] for idx, item in enumerate(curr_mask_pos_token0)]
            hidden_token0 = torch.stack(hidden_token0, dim=0)
            emb_token0 = self.model.lm_head.layer_norm(self.model.lm_head.dense(hidden_token0))
            logits_token0 = self.model.lm_head.decoder(emb_token0)
            prob_token0 = torch.softmax(logits_token0, dim=1)
            loss_token0 = self.alpha * torch.sum(F.cross_entropy(logits_token0, curr_label_token0, reduction="none"))
            #loss_token0 = -torch.sum(prob_token0)
            self.curr_embedding = None


            outputs_token1 = self.model.roberta(input_ids=xs_token1, attention_mask=masks_token1, output_hidden_states=True)
            hidden_token1 = [outputs_token1[0][idx, item, :] for idx, item in enumerate(curr_mask_pos_token1)]
            hidden_token1 = torch.stack(hidden_token1, dim=0)
            emb_token1 = self.model.lm_head.layer_norm(self.model.lm_head.dense(hidden_token1))
            logits_token1 = self.model.lm_head.decoder(emb_token1)
            prob_token1 = torch.softmax(logits_token1, dim=1)
            loss_token1 = self.alpha * torch.sum(F.cross_entropy(logits_token1, curr_label_token1, reduction="none"))
            # loss_token1 = -torch.sum(prob_token1)
            loss_token1.backward()
            grads_tmp_token1 = self.curr_embedding.grad.clone().cpu().numpy()
            grads_tmp_token1 = [item[curr_trigger_offset[idx]:curr_trigger_offset[idx] + 1]
                                for idx, item in enumerate(grads_tmp_token1)]
            grads_token1 = []
            pad_width = ((0, len(self.trigger)-1), (0, 0))
            for arr in grads_tmp_token1:
                padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
                grads_token1.append(padded_arr)

            self.curr_embedding.grad.zero_()
            self.curr_embedding = None

            outputs_token2 = self.model.roberta(input_ids=xs_token2, attention_mask=masks_token2, output_hidden_states=True)
            hidden_token2 = [outputs_token2[0][idx, item, :] for idx, item in enumerate(curr_mask_pos_token2)]
            hidden_token2 = torch.stack(hidden_token2, dim=0)
            emb_token2 = self.model.lm_head.layer_norm(self.model.lm_head.dense(hidden_token2))
            logits_token2 = self.model.lm_head.decoder(emb_token2)
            prob_token2 = torch.softmax(logits_token2, dim=1)
            loss_token2 = self.alpha * torch.sum(F.cross_entropy(logits_token2, curr_label_token2, reduction="none"))
            # loss_token2 = -torch.sum(prob_token2)
            loss_token2.backward()
            grads_tmp_token2 = self.curr_embedding.grad.clone().cpu().numpy()
            grads_tmp_token2 = [item[curr_trigger_offset[idx]:curr_trigger_offset[idx] + 2]
                                for idx, item in enumerate(grads_tmp_token2)]
            grads_token2 = []
            pad_width = ((0, len(self.trigger)-2), (0, 0))
            for arr in grads_tmp_token2:
                padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
                grads_token2.append(padded_arr)

            self.curr_embedding.grad.zero_()
            self.curr_embedding = None
            
            
                       
            outputs_token3 = self.model.roberta(input_ids=xs_token3, attention_mask=masks_token3, output_hidden_states=True)
            hidden_token3 = [outputs_token3[0][idx, item, :] for idx, item in enumerate(curr_mask_pos_token3)]
            hidden_token3 = torch.stack(hidden_token3, dim=0)
            emb_token3 = self.model.lm_head.layer_norm(self.model.lm_head.dense(hidden_token3))
            logits_token3 = self.model.lm_head.decoder(emb_token3)
            prob_token3 = torch.softmax(logits_token3, dim=1)
            loss_token3 = self.alpha * torch.sum(F.cross_entropy(logits_token3, curr_label_token3, reduction="none"))
            # loss_token1 = -torch.sum(prob_token1)
            loss_token3.backward()
            grads_tmp_token3 = self.curr_embedding.grad.clone().cpu().numpy()
            grads_tmp_token3 = [item[curr_trigger_offset[idx]:curr_trigger_offset[idx] + 3]
                                for idx, item in enumerate(grads_tmp_token3)]
            grads_token3 = []
            pad_width = ((0, len(self.trigger)-3), (0, 0))
            for arr in grads_tmp_token3:
                padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
                grads_token3.append(padded_arr)

            self.curr_embedding.grad.zero_()
            self.curr_embedding = None
            
            
            outputs_token4 = self.model.roberta(input_ids=xs_token4, attention_mask=masks_token4, output_hidden_states=True)
            hidden_token4 = [outputs_token4[0][idx, item, :] for idx, item in enumerate(curr_mask_pos_token4)]
            hidden_token4 = torch.stack(hidden_token4, dim=0)
            emb_token4 = self.model.lm_head.layer_norm(self.model.lm_head.dense(hidden_token4))
            logits_token4 = self.model.lm_head.decoder(emb_token4)
            prob_token4 = torch.softmax(logits_token4, dim=1)
            loss_token4 = self.alpha * torch.sum(F.cross_entropy(logits_token4, curr_label_token4, reduction="none"))
            loss_token4.backward()
            grads_tmp_token4 = self.curr_embedding.grad.clone().cpu().numpy()
            grads_tmp_token4 = [item[curr_trigger_offset[idx]:curr_trigger_offset[idx] + 4]
                                for idx, item in enumerate(grads_tmp_token4)]
            grads_token4 = []
            pad_width = ((0, len(self.trigger)-4), (0, 0))
            for arr in grads_tmp_token4:
                padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
                grads_token4.append(padded_arr)
            self.curr_embedding.grad.zero_()
            self.curr_embedding = None

                    
            outputs = self.model.roberta(input_ids=xs, attention_mask=masks, output_hidden_states=True)
            hidden = [outputs[0][idx, item, :] for idx, item in enumerate(curr_mask_pos)]
            hidden = torch.stack(hidden, dim=0)
            emb = self.model.lm_head.layer_norm(self.model.lm_head.dense(hidden))
            logits = self.model.lm_head.decoder(emb)
            prob = torch.softmax(logits, dim=1)


            # if self.loss_type == "embdis":
            #     loss = emb - emb_no_trigger
            #     loss = -torch.sqrt((loss * loss).sum(dim=1)).sum()
            #     loss.backward()
            # elif self.loss_type in ["prob", "prob_sq"]:
            #     loss = torch.gather(prob, dim=1, index=curr_label.unsqueeze(1)).squeeze(1)
            #     if self.loss_type == "prob_sq":
            #         loss = loss * loss
            #     loss = loss.sum()
            #     loss.backward()
            # elif self.loss_type in ["mprob", "mprob_sq"]:
            #     loss = prob.max(dim=1)[0]
            #     if self.loss_type == "mprob_sq":
            #         loss = loss * loss
            #     loss = loss.sum()
            #     loss.backward()
            # elif self.loss_type in ["ce"]:

            loss_1 = F.cross_entropy(logits, curr_label, reduction="none")
            loss_1 = -loss_1.sum()
            loss_1.backward()
            loss_2 = loss_token0 + loss_token1 + loss_token2 + loss_token3 + loss_token4
            loss = loss_1 + loss_2
            # else:
                # assert 0
            overall_loss += loss.detach().cpu().numpy()
            

            result.append(prob.detach().cpu())
            grads_tmp_1 = self.curr_embedding.grad.clone().cpu().numpy()
            grads_tmp_1 = [item[curr_trigger_offset[idx]:curr_trigger_offset[idx] + len(self.trigger)]
                         for idx, item in enumerate(grads_tmp_1)]
                        
            grads_tmp = list(np.sum([grads_tmp_1,grads_token1,grads_token2,grads_token3,grads_token4],axis=0)) 
            result_grad.append(grads_tmp)

            self.curr_embedding.grad.zero_()
            self.curr_embedding = None
            
            del hidden
            del emb
            del emb_no_trigger
            del prob
            del loss
            del hidden_token0
            del emb_token0
            del prob_token0
            del loss_token0
            del loss_1
            del hidden_token1
            del emb_token1
            del prob_token1
            del loss_token1
            del grads_tmp_token1
            del grads_token1
            del hidden_token2
            del emb_token2
            del prob_token2
            del loss_token2
            del loss_2
            del grads_tmp_token2
            del grads_token2
            del hidden_token3
            del emb_token3
            del prob_token3
            del loss_token3
            del grads_tmp_token3
            del grads_token3
            del hidden_token4
            del emb_token4
            del prob_token4
            del loss_token4
            del grads_token4
            del grads_tmp_token4
            

        result = np.concatenate(result, axis=0)
        result_grad = np.concatenate(result_grad, axis=0)
        if return_loss:
            return result, result_grad, overall_loss
        else:
            return result, result_grad

''' 
    def predict_link(self, input_, pos=0):
        sen_list_l = []
        mask_pos_l = []
        labels_l   = []
        
        for text in input_:
            text = [tok.strip() for tok in text]
            trigger_index = text.index("<trigger>")
            if pos == 0:
                text_l= text[:trigger_index] + ["<mask>"]
            else:
                text_l= text[:trigger_index] + self.trigger[0:pos] + ["<mask>"]
            mask_pos_tmp_l = text_l.index("<mask>")
            sen_list_l.append(text_l)
            mask_pos_l.append(mask_pos_tmp_l+1)
            labels_l.append(self.tokenizer.convert_tokens_to_ids(self.trigger[pos]))

        sent_lens_l = [len(sen) for sen in sen_list_l]
        batch_len_l = max(sent_lens_l) + 2

        attentions_l = np.array([[1] * (len(sen) + 2) + [0] * (batch_len_l - 2 - len(sen))
                                    for sen in sen_list_l], dtype='int64')
        sen_list_l = [self.tokenizer.convert_tokens_to_ids(sen) for sen in sen_list_l]
        input_ids_l = np.array([
            [self.tokenizer.cls_token_id] + sen + [self.tokenizer.sep_token_id]
            + [self.tokenizer.pad_token_id] * (batch_len_l - 2 - len(sen)) for sen in sen_list_l], dtype='int64')
        labels_l = torch.LongTensor(labels_l).to(self.device)

        overall_loss_l = 0
        for i in range((len(sen_list_l) + self.batch_size - 1) // self.batch_size):
            curr_input_ids_l = input_ids_l[i * self.batch_size: (i + 1) * self.batch_size]
            curr_mask_l = attentions_l[i * self.batch_size: (i + 1) * self.batch_size]
            curr_label_l = labels_l[i * self.batch_size: (i + 1) * self.batch_size]
            curr_mask_pos_l = mask_pos_l[i * self.batch_size: (i + 1) * self.batch_size]

            xs_l = torch.from_numpy(curr_input_ids_l).long().to(self.device)
            masks_l = torch.from_numpy(curr_mask_l).long().to(self.device)
            outputs_l = self.model.roberta(input_ids=xs_l, attention_mask=masks_l, output_hidden_states=True)
            hidden_l = [outputs_l[0][idx, item, :] for idx, item in enumerate(curr_mask_pos_l)]
            hidden_l = torch.stack(hidden_l, dim=0)
            emb_l = self.model.lm_head.layer_norm(self.model.lm_head.dense(hidden_l))
            logits_l = self.model.lm_head.decoder(emb_l)
            prob_l = torch.softmax(logits_l, dim=1)

            #loss_l = F.cross_entropy(logits_l, curr_label_l, reduction="none")
            loss_l = prob_l
            loss_l = loss_l.sum()
            loss_l.backward()

            overall_loss_l += loss_l.detach().cpu().numpy()

            del hidden_l
            del emb_l
            del prob_l
            del loss_l

        return overall_loss_l
'''
