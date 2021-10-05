import pdb
from transformers import BertTokenizer, BertConfig, BertModel
import torch
import numpy as np
import pickle
import argparse


def aggregate_reps(reps_list):
    reps = torch.zeros([len(reps_list), 768])
    for i, wrep in enumerate(reps_list):
        w, rep = wrep
        reps[i] = rep
    if len(reps) > 1:
        reps = torch.mean(reps, axis=0)
    try:
        reps = reps.view(768)
    except RuntimeError:
        pdb.set_trace()
    return reps

def check_correct_token_mapping(bert_tokenized_sentence, bert_positions, target_word, tokenizer):
    tokenized_word = list(tokenizer.tokenize(target_word))
    bert_token = [bert_tokenized_sentence[p] for p in bert_positions]
    if bert_token == tokenized_word:
        return True
    return False


def tokenize_and_map_idcs(sentence, tokenizer):
    map_ori_to_bert = []
    tok_sent = ['[CLS]']
    for orig_token in sentence.split():
        current_tokens_bert_idx = [len(tok_sent)]
        bert_token = tokenizer.tokenize(orig_token) # tokenize
        tok_sent.extend(bert_token) # add to my new tokens
        if len(bert_token) > 1: # if the new token has been 'wordpieced'
            extra = len(bert_token) - 1
            for i in range(extra):
                current_tokens_bert_idx.append(current_tokens_bert_idx[-1]+1) # list of new positions of the target word in the new tokenization
        map_ori_to_bert.append(tuple(current_tokens_bert_idx))

    tok_sent.append('[SEP]')
    return tok_sent, map_ori_to_bert


def extract_representations(tokenized_sentences, infos, tokenizer, model_name, maxlen):
    reps_by_type = dict()
    reps_by_type_adj = dict()
    config = BertConfig.from_pretrained(model_name, output_hidden_states=True, max_len=maxlen)
    model = BertModel.from_pretrained(model_name, config=config)

    model.eval()
    with torch.no_grad():
        for tok_sent, info in zip(tokenized_sentences, infos):
            input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tok_sent)])
            input_ids.to(device)
            inputs = {'input_ids': input_ids}
            outputs = model(**inputs)
            hidden_states = outputs[2]
            infotup = tuple(info.items())

            if infotup not in reps_by_type:
                reps_by_type[infotup] = dict()
                reps_by_type_adj[infotup] = dict()

            for laynum in range(0, toplayer):
                reps_by_type[infotup][laynum] = []
                reps_by_type_adj[infotup][laynum] = []
                for i, w in enumerate(tok_sent):
                    if i in info['bertposition']:
                        reps_by_type[infotup][laynum].append((w, hidden_states[laynum][0][i].cpu()))
                    if "bertposition_adj" in info and i in info["bertposition_adj"]:
                        reps_by_type_adj[infotup][laynum].append((w, hidden_states[laynum][0][i].cpu()))

    return reps_by_type, reps_by_type_adj



def load_dataset(HVD_sentences, maxlen):
    infos = []
    sentences = []
    tokenized_sentences = []

    for an in HVD_sentences:
        noun, adjective = an
        instance = HVD_sentences[an][0]
        sentences.append(" ".join(instance["sentence1"]))
        sentences.append(" ".join(instance["sentence2"]))
        info = dict()
        info['id'] = noun + "_" + adjective + "-1"
        info['position'] = instance['position1']
        info['word'] = noun
        info["actual_word"] = instance["sentence1"][instance["position1"]]
        info['adjective'] = adjective
        infos.append(info)
        info = dict()
        info['id'] = noun + "_" + adjective + "-2"
        info['position'] = instance['position2']
        info['word'] = noun
        info['actual_word'] = instance["sentence2"][instance["position2"]]
        info['adjective'] = adjective
        infos.append(info)

    for sent, info in zip(sentences, infos):
        tok_sent, mapp = tokenize_and_map_idcs(sent, tokenizer)
        info['bertposition'] = mapp[int(info['position'])]
        ### verify that we have the right index. Ignore plural forms
        if not check_correct_token_mapping(tok_sent, info['bertposition'], info['actual_word'], tokenizer) and info["position"] != 0:
            print("Index mismatch")
            pdb.set_trace()
        if info['bertposition'][-1] > maxlen:
            print("position of target token exceeds maximum sequence length")
            pdb.set_trace()
        if info['id'].endswith('-2'):
            if info["position"] == 0: # if we used the plural template (raspberries are edible), the adjective is at the end of the sentence
                info["bertposition_adj"] = mapp[len(sent.split())-2]
            else: # otherwise it's right before the noun
                info['bertposition_adj'] = mapp[int(info['position'])-1]
        tokenized_sentences.append(tok_sent)

    return tokenized_sentences, infos


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str)
    args = parser.parse_args()
    toplayer = 13
    maxlen = 512
    do_lower_case = True if "uncased" in args.model_name_or_path else False
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=do_lower_case)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.manual_seed(0)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dsname = "HVD_sentences.pkl"
    HVD_sentences = pickle.load(open(dsname, "rb"))

    tokenized_sentences, infos = load_dataset(HVD_sentences,maxlen=maxlen)

    reps_by_type, reps_by_type_adj = extract_representations(tokenized_sentences, infos, tokenizer, args.model_name_or_path, maxlen)

    new_data = dict()

    for AN in HVD_sentences:
        for i, sentence in enumerate(HVD_sentences[(AN[0], AN[1])][:1]):
            idi = AN[0] + "_" + AN[1]
            new_data[idi] = dict()
            for k in ["reps_noun1", "reps_noun2","reps_adj2"]:
                new_data[idi][k] = dict()

    for infotup in reps_by_type:
        infodi = dict(infotup)
        idi = infodi["id"]
        noun = infodi["word"]
        adjective = infodi["adjective"]
        if (noun, adjective) not in new_data:
            new_data[(noun, adjective)] = dict()
            new_data[(noun, adjective)]["reps_noun1"] = dict()
            new_data[(noun, adjective)]["reps_noun2"] = dict()
            new_data[(noun, adjective)]["reps_adj2"] = dict()

        for laynum in reps_by_type[infotup]:
            representation = aggregate_reps(reps_by_type[infotup][laynum]) # noun representation
            if idi.split("-")[-1] == "1":  # if it's the first sentence (no adjective)
                new_data[(noun, adjective)]["reps_noun1"][laynum] = representation.cpu()
            elif idi.split("-")[-1] == "2":
                new_data[(noun, adjective)]["reps_noun2"][laynum] = representation.cpu()
                new_data[(noun, adjective)]["reps_adj2"][laynum] = aggregate_reps(reps_by_type_adj[infotup][laynum]).cpu()


    pickle.dump(new_data, open("HVD_BERT_representations.pkl", "wb"))

