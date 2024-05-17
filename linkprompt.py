import argparse
import datetime
import json
import os
import datasets
import numpy as np
import pandas
import torch
import tqdm
import nltk
from nltk.tokenize import sent_tokenize
from utils import RobertaClassifier

datasets.set_caching_enabled(False)

exp_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

output_dir_root = "triggers/"

nltk.download('punkt')

def dataset_mapping_wiki(item, tokenizer):
    """Process wiki data. For each sentence, mask one word and insert <trigger> near the <mask>."""
    if item["text"].startswith("= ") or len(item["text"].split()) < 20:
        return None

    list_of_sents = sent_tokenize(item["text"]) 
    st = np.random.randint(len(list_of_sents)) 
    ed = st + 3 + np.random.randint(5) 
    text = " ".join(list_of_sents[st:ed]) 
    toks = tokenizer.tokenize(text.strip())
    if len(toks) < 20:
        return None

    toks = toks[-100:]
    mask_pos = np.random.choice(int(0.1 * len(toks)))
    mask_pos = len(toks) - mask_pos - 1
    trigger_pos = max(0, mask_pos - np.random.randint(5))
    label = tokenizer.vocab[toks[mask_pos]]
    toks[mask_pos] = "<mask>"
    toks = toks[:trigger_pos] + ["<trigger>"] + toks[trigger_pos:]
    return {
        "x": tokenizer.convert_tokens_to_string(toks),
        "y": label
    } 


def search_triggers_on_pretrained_lm(victim, dataset, tokenizer, epoch, batch_size,
                                     trigger_len, used_tokens, beam_size=5, bin_id=0):
    word2id = victim.word2id
    embedding = victim.embedding
    id2word = {v: k for k, v in word2id.items()}

    def get_candidates(gradient, current_word_ids, pos):
        args = (embedding - embedding[current_word_ids[pos]]).dot(gradient.T).argsort()
        ret = []
        if pos == 0:
            for idx in args:
                if idx == current_word_ids[pos]:
                    continue
                if id2word[idx] in used_tokens:
                    continue
                if id2word[idx][0] == "Ġ":
                    ret.append(id2word[idx])
                    if len(ret) == beam_size:
                        break
        else:
            for idx in args:
                if idx == current_word_ids[pos]:
                    continue
                if id2word[idx] in used_tokens:
                    continue
                word = id2word[idx]
                # ignore special tokens
                if len(word) == 0 or (word[0] == "<" and word[-1] == ">"):
                    continue
                tmp = current_word_ids[:pos] + [idx] + current_word_ids[pos + 1:]
                tmp_detok = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tmp))
                tmp_rec = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tmp_detok))
                if len(tmp_rec) != len(tmp) or tmp[pos] != tmp_rec[pos]:
                    continue
                ret.append(id2word[idx])
                if len(ret) == beam_size:
                    break

        if len(ret) != beam_size:
            print("warning", current_word_ids)
        return ret

    curr_trigger_list = []
    for i in range(beam_size):
        trigger_tmp = []
        for j in range(trigger_len):
            tmp = np.random.choice(list(word2id.keys()))
            while len(tmp) == 0 or tmp[0] != "Ġ":
                tmp = np.random.choice(list(word2id.keys()))
            trigger_tmp.append(tmp)
        curr_trigger_list.append(trigger_tmp)

    for epoch_idx in range(epoch):
        for num_iter in tqdm.tqdm(range((len(dataset) + batch_size - 1) // batch_size),
                                  desc="Trigger %d Epoch %d: " % (bin_id, epoch_idx)):
            cnt = num_iter * batch_size
            batch = dataset[cnt: cnt + batch_size]

            x = [tokenizer.tokenize(" " + sent) for sent in batch["x"]]
            y = batch["y"]

            nw_beams = []
            for item in curr_trigger_list:
                victim.set_trigger(item)
                _, _, loss = victim.predict(x, labels=y, return_loss=True)
                nw_beams.append((item, loss))
            print("=======")
            print(nw_beams)

            for i in range(trigger_len):
                beams = nw_beams[:]
                for trigger, _ in beams:
                    victim.set_trigger(trigger)
                    grad = victim.get_grad(x, labels=y)[1]
                    candidates_words = get_candidates(grad[:, i, :].mean(axis=0),
                                                      tokenizer.convert_tokens_to_ids(trigger), pos=i)

                    for cw in candidates_words:
                        if cw in trigger[:i]:
                            #print("same token")
                            continue
                        tt = trigger[:i] + [cw] + trigger[i + 1:] 
                        
                        duplicate = False
                        for trigger_tmp, loss in nw_beams:
                            if trigger_tmp == tt:
                                duplicate = True
                                break
                        if duplicate:
                            continue

                        victim.set_trigger(tt)
                        _, _, loss= victim.predict(x, labels=y, return_loss=True)
                        nw_beams.append((tt, loss))
                nw_beams = sorted(nw_beams, key=lambda x: x[1])[:beam_size]
                print(nw_beams[:3])
            curr_trigger_list = [item[0] for item in nw_beams]

    x = [tokenizer.tokenize(" " + sent) for sent in dataset["x"]]
    y = dataset["y"]
    nw_beams = []
    for trigger in curr_trigger_list:
        victim.set_trigger(trigger)
        _, _, loss = victim.predict(x, labels=y, return_loss=True)
        nw_beams.append((trigger, loss))
    nw_beams = sorted(nw_beams, key=lambda x: x[1])[:beam_size]
    return [item[0] for item in nw_beams]


def parse_args():
    parser = argparse.ArgumentParser("""Search for adversarial triggers on RoBERTa-large.""",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--trigger_len", type=int, default=5,
                        help="The length of the trigger.")
    parser.add_argument("--num_triggers", type=int, default=3, help="Number of triggers to be found.")
    parser.add_argument("--subsample_size", type=int, default=1536,
                        help="Subsample the dataset. The sub-sampled dataset will be evenly splitted to search for "
                        "each trigger. \n"
                        "By default, a total of 1536 sentences will be splitted to three 512-sentence "
                        "subsets to find 3 triggers.")
    parser.add_argument("--batch_size", type=int, default=16, help="Trigger search batch size.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to search each trigger.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--alpha", type=float, default=0, help="The weight of the loss of triggers segment similarity.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id.")
    return vars(parser.parse_args())


def main():
    meta = parse_args()

    print("Load roberta large.")
    np.random.seed(meta["seed"])
    torch.manual_seed(meta["seed"])
    victim = RobertaClassifier(torch.device("cuda:%d" % meta["gpu"]), trigger_len=meta["trigger_len"], alpha=meta["alpha"])
    print("Load dataset.")
    meta["dataset"] = dataset_name = "wikitext"
    dataset_subset = "wikitext-2-raw-v1"
    trainset_raw = list(datasets.load_dataset(dataset_name, dataset_subset, split="train"))
    np.random.shuffle(list(trainset_raw))
    trainset = []
    for item in trainset_raw:
        item_tmp = dataset_mapping_wiki(item, tokenizer=victim.tokenizer)
        if item_tmp is not None:
            trainset.append(item_tmp)
            if len(trainset) == meta["subsample_size"]:
                break

    assert len(trainset) == meta["subsample_size"]

    used_tokens = []

    bin_size = len(trainset) // meta["num_triggers"]
    triggers = []
    for bin_id in range(meta["num_triggers"]):
        bin_data = trainset[bin_size * bin_id:bin_size * (bin_id + 1)]

        trigger_tmp = search_triggers_on_pretrained_lm(
            victim, datasets.Dataset.from_pandas(pandas.DataFrame(bin_data)), victim.tokenizer,
            epoch=meta["num_epochs"], batch_size=meta["batch_size"],
            trigger_len=meta["trigger_len"], used_tokens=used_tokens, bin_id=bin_id)
        triggers.append(trigger_tmp[0])
        used_tokens.extend(trigger_tmp[0])

    print(triggers)
    meta["triggers"] = triggers
    os.makedirs(output_dir_root, exist_ok=True)
    with open(output_dir_root + "len{len}_alpha{alpha}_seed{seed}_{exp_id}.json".format(
            len=meta["trigger_len"], alpha=meta["alpha"],
            seed=meta["seed"], exp_id=exp_id), "w") as f:
        json.dump(meta, f, indent=4)


if __name__ == "__main__":
    main()
