'''
Copied and adapted from an example script in https://github.com/huggingface/
(not available anymore)
'''
import os
import argparse
import torch
from torch import nn
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, BertPreTrainedModel, BertModel
from torch.utils.data import DataLoader, SequentialSampler
import pdb
from transformers import AdamW, get_linear_schedule_with_warmup
import logging
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from copy import copy
import random
import pickle
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss
from cv_classif_mcrae import load_folds, organize_data_byfold, ANpair

seed = torch.manual_seed(1)
logger = logging.getLogger(__name__)

random.seed(0)



'''
In AddoneDataset, the comparison can be:
noun_noun: n1, n2 (N_s_n and N_s_an)
noun_adj: n1, a2 (N_s_n and A_s_an)
noun_adjnoun: n1, (a2+nf2) (N_s_n and A_s_an+N_s_an)
'''

class BertForAddOneClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForAddOneClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)
        self.device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.init_weights()

    def forward(self, input_ids, input_target_indices, input_target_indices_noun2=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        # First, run the output through bert normally:
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        last_hidden_state = outputs[0]

        if input_target_indices_noun2 is None:
            pooled_output_x1, pooled_output_x2 = get_tw_representations(input_target_indices,
                                                                        last_hidden_state)
        elif input_target_indices_noun2 is not None:
            noun1 = input_target_indices[:, 0]
            adj = input_target_indices[:, 1]
            pooled_output_x1, pooled_output_x2 = get_tw_representations_proto_adjnoun(noun1, adj,
                                                                                      input_target_indices_noun2,
                                                                                      last_hidden_state)

        pooled_output = torch.cat((pooled_output_x1, pooled_output_x2), 1)

        pooled_output = self.dropout(pooled_output)
        pooled_output = pooled_output.to(self.device2)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]


        if labels is not None:
            if self.num_labels == 1: # regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else: # classification
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


MODEL_CLASSES = {'bert': BertForSequenceClassification, 'bert-bytoken': BertForAddOneClassification}

def get_tw_representations(input_target_indices, last_hidden_state):
    pooled_output_x1 = torch.zeros([last_hidden_state.shape[0], last_hidden_state.shape[2]])
    pooled_output_x2 = torch.zeros([last_hidden_state.shape[0], last_hidden_state.shape[2]])

    for i, (tw_indices, sequence) in enumerate(zip(input_target_indices, last_hidden_state)):
        tw1_vecs = torch.index_select(sequence, 0, tw_indices[0][
            torch.nonzero(tw_indices[0]).flatten()].long())
        tw2_vecs = torch.index_select(sequence, 0,
                                      tw_indices[1][torch.nonzero(tw_indices[1]).flatten()].long())
        tw1_vec = torch.mean(tw1_vecs, axis=0)
        tw2_vec = torch.mean(tw2_vecs, axis=0)
        pooled_output_x1[i] = tw1_vec
        pooled_output_x2[i] = tw2_vec

    return pooled_output_x1, pooled_output_x2


def position_aware_tokenization(sentence_pairs, tw_positions, labels, tokenizer, MAX_LEN):
    max_numb_indices = 0

    new_sentences = []
    new_tw_positions = []
    new_token_type_ids = []
    new_labels = []

    for sents, twp, label in zip(sentence_pairs, tw_positions, labels):
        token_type_ids = [0]
        s1, s2 = sents  # s1 and s2: sentence1 and sentence2
        orig_to_tok_map_s1 = []
        bert_tokens = []
        bert_tokens.append("[CLS]")
        # tokenize sentence word by word and update position information according to number of wordpieces
        for orig_token in s1.split():
            current_tokens_bert_idx = [len(bert_tokens)]
            new_token = tokenizer.tokenize(orig_token)  # tokenize
            bert_tokens.extend(new_token)  # add to my new tokens
            token_type_ids.extend([0] * len(new_token))

            if len(new_token) > 1:  # if the new token has been 'wordpieced'
                extra = len(new_token) - 1
                for i in range(extra):
                    current_tokens_bert_idx.append(current_tokens_bert_idx[-1] + 1)  # list of new positions of the target word in the new tokenization
            orig_to_tok_map_s1.append(current_tokens_bert_idx)

        new_position_1 = orig_to_tok_map_s1[twp["position1"]]

        bert_tokens.append("[SEP]")
        token_type_ids.append(0)

        tokenized_s2 = []
        orig_to_tok_map_s2 = []
        # now tokenize 2nd sentence word by word
        for orig_token in s2.split():
            current_tokens_bert_idx = [len(tokenized_s2)]
            new_token = tokenizer.tokenize(orig_token)
            tokenized_s2.extend(new_token)
            token_type_ids.extend([1] * len(new_token))
            if len(new_token) > 1:
                extra = len(new_token) - 1
                for i in range(extra):
                    current_tokens_bert_idx.append(current_tokens_bert_idx[-1] + 1)
            orig_to_tok_map_s2.append(current_tokens_bert_idx)

        try:
            new_position_2_almost = orig_to_tok_map_s2[twp["position2"]]
        except IndexError:
            pdb.set_trace()

        new_position_2 = [p + len(bert_tokens) for p in new_position_2_almost]
        bert_tokens.extend(tokenized_s2)
        bert_tokens.append("[SEP]")
        token_type_ids.append(1)

        # If the target word is out of reach (512 tokens), ignore this example
        if new_position_1[-1] >= MAX_LEN or new_position_2[-1] >= MAX_LEN:
            print("omitted one instance!")
            continue
        new_sentences.append(bert_tokens)
        new_labels.append(label)

        new_twp = copy(twp)
        new_twp["position1"] = new_position_1
        new_twp["position2"] = new_position_2
        new_tw_positions.append(new_twp)

        if len(new_position_1) > max_numb_indices:
            max_numb_indices = len(new_position_1)
        elif len(new_position_2) > max_numb_indices:
            max_numb_indices = len(new_position_2)

        new_token_type_ids.append(token_type_ids)
    return new_sentences, new_tw_positions, new_labels, new_token_type_ids, max_numb_indices


def load_mcrae_data(dataset, subset, test_fold):
    ori_data = []
    if dataset == "HVD":
        sentences_fn = "HVD_sentences.pkl"
        labels_fn = "HVD_labels.csv"

    with open(labels_fn) as f:
        for l in f:
            n, a, label = l.strip().split("\t")
            instance = ANpair(n, a, "HVD")
            instance.typ = label
            ori_data.append(instance)

    fold_info = load_folds()

    data = organize_data_byfold(fold_info, ori_data, test_fold)
    data = data[subset]
    sentences = pickle.load(open(sentences_fn, "rb"))

    return data, sentences


class McRaeDataset(Dataset):
    def __init__(self, dataset, subset, tokenizer, model="bert", maxlen=512, comparison="noun_adj", test_fold=1):
        data, rep_sentences = load_mcrae_data(dataset, subset, test_fold=test_fold)
        self.labels = []
        input_ids = []
        token_type_ids = []
        self.model = model
        self.comparison = comparison
        sentence_pairs = []
        tw_positions = []
        secondnoun_and_position = []
        self.model = model
        limit=1
        for instance in data:
            an = (instance.noun, instance.adjective)
            for sentence in rep_sentences[an][:limit]: 
                if "bert" in self.model and self.model != "bert-bytoken":
                    tokenized_sequence = (tokenizer.tokenize(" ".join(sentence['sentence1'])), tokenizer.tokenize(" ".join(sentence['sentence2'])))                    
                    seq1 = tokenizer.convert_tokens_to_ids(tokenized_sequence[0])
                    seq2 = tokenizer.convert_tokens_to_ids(tokenized_sequence[1])
                    one_inputids = torch.Tensor(tokenizer.build_inputs_with_special_tokens(seq1, seq2))
                    one_ttids = torch.Tensor(tokenizer.create_token_type_ids_from_sequences(seq1, seq2))
                    token_type_ids.append(one_ttids)
                    input_ids.append(one_inputids)
                elif self.model == "bert-bytoken":
                    ##### in noun_noun, default noun positions
                    ##### in noun_adj, position 1 and position2-1 (to get the noun in the first sentence and the adjective in the second)
                    position1 = int(sentence['position1'])
                    position2 = int(sentence['position2'])

                    sentence_pairs.append((" ".join(sentence['sentence1']), " ".join(sentence['sentence2'])))
                    if comparison == "noun_noun":
                        tw_positions.append({"position1": position1, 'position2': position2})
                    elif comparison in ["noun_adj", "noun_adjnoun"]:
                        if position2 == 0: # in sentences of the type "raspberries are edible", the adjective will be at position2 +2
                            adjposition = position2 + 2
                        else:
                            adjposition = position2 -1
                        tw_positions.append({"position1": position1, 'position2': adjposition})
                        if comparison == "noun_adjnoun":
                            secondnoun_and_position.append((sentence['sentence2'][position2], position2))

                if instance.typ in ["YES","NO"]:
                    label = 1 if instance.typ == 'YES' else 0
                else:
                    label = float(instance.typ)
                self.labels.append(label)

        if self.model == "bert-bytoken":
            new_sentences, new_tw_positions, new_labels, new_token_type_ids, max_numb_indices = position_aware_tokenization(sentence_pairs, tw_positions, self.labels, tokenizer, maxlen)
            if comparison == "noun_adjnoun":
                tw_positions_noun_to_combine = []
                for bertindices, noun_position in zip(new_tw_positions, secondnoun_and_position):
                    noun, position = noun_position
                    tokenized_noun = tokenizer.tokenize(noun)
                    last_adj_bert_index = bertindices['position2'][-1]
                    noun_bert_indices = []
                    i = 1
                    for _ in range(len(tokenized_noun)):
                        noun_bert_indices.append(last_adj_bert_index+i)
                        i+=1
                    tw_positions_noun_to_combine.append(noun_bert_indices)
                    max_noun_numb_indices = max([len(x) for x in tw_positions_noun_to_combine])
                    self.twnoun_tensor = torch.tensor(pad_sequences(tw_positions_noun_to_combine, maxlen=max_noun_numb_indices, padding="post"))

            input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in new_sentences], maxlen=maxlen, dtype="long", truncating="post", padding="post")
            token_type_ids = pad_sequences(new_token_type_ids, maxlen=maxlen, dtype="long", truncating="post", padding="post", value=0)
            tw1l = [tw["position1"] for tw in new_tw_positions]
            tw2l = [tw["position2"] for tw in new_tw_positions]
            tw1_tensor = torch.tensor(pad_sequences(tw1l, maxlen=max_numb_indices, padding="post"))
            tw2_tensor = torch.tensor(pad_sequences(tw2l, maxlen=max_numb_indices, padding="post"))
            self.tokenized_positions = torch.stack([tw1_tensor, tw2_tensor], 1)

        elif self.model == "bert":
            input_ids = pad_sequences(input_ids, maxlen=maxlen, dtype="long", truncating="post", padding="post")
            token_type_ids = pad_sequences(token_type_ids, maxlen=maxlen, dtype="long", truncating="post", padding="post",value=0)

        attention_masks = [[float(i != 0) for i in ii] for ii in input_ids]

        self.input_tensor = torch.tensor(input_ids)
        self.token_type_ids_tensor = torch.tensor(token_type_ids)
        self.attentionmask_tensor = torch.tensor(attention_masks)
        self.label_tensor = torch.tensor(self.labels)


    def __getitem__(self, index):
        sentence_x = self.input_tensor[index]
        label_y = self.label_tensor[index]
        attention_mask = self.attentionmask_tensor[index]
        token_type_id = self.token_type_ids_tensor[index]
        if self.model != "bert-bytoken":
            return sentence_x, label_y, token_type_id, attention_mask
        else:
            position = self.tokenized_positions[index]
            return sentence_x, label_y, token_type_id, attention_mask, position


    def __len__(self):
        return len(self.input_tensor)


class AddOneDataset(Dataset):
        def __init__(self, subset, tokenizer, model="bert", maxlen=256, comparison="noun_noun"):
                self.comparison = comparison

                data = pickle.load(open("add_one.labels_and_positions." + subset + ".pkl", "rb")) # subset is one of ["train", "test","dev"] 
                # tokenize data
                self.labels = []
                input_ids = []
                token_type_ids = []
                sentence_pairs = []
                tw_positions = []
                secondnoun_and_position = []
                self.model = model
                if subset == "train":
                        random.shuffle(data)
                for instance in data:
                        label = 1 if instance['label'] == 'YES' else 0
                        if "bert" in self.model and self.model != "bert-bytoken":
                                tokenized_sequence = (tokenizer.tokenize(instance['sentence1']), tokenizer.tokenize(instance['sentence2']))
                                seq1 = tokenizer.convert_tokens_to_ids(tokenized_sequence[0])
                                seq2 = tokenizer.convert_tokens_to_ids(tokenized_sequence[1])                           
                                one_inputids = torch.Tensor(tokenizer.build_inputs_with_special_tokens(seq1, seq2))
                                one_ttids = torch.Tensor(tokenizer.create_token_type_ids_from_sequences(seq1, seq2))
                                token_type_ids.append(one_ttids)
                                input_ids.append(one_inputids)

                        elif self.model == "bert-bytoken":
                            sentence_pairs.append((instance['sentence1'], instance['sentence2']))
                            if comparison == "noun_noun":
                                tw_positions.append({"position1": instance['position1'], 'position2': instance['position2']})
                            elif comparison in ["noun_adj", "noun_adjnoun"]:
                                tw_positions.append({"position1": instance['position1'], 'position2': instance['position2'] - 1})
                                if comparison == "noun_adjnoun":
                                    secondnoun_and_position.append((instance['sentence2'].split()[instance['position2']], instance['position2']))
                        self.labels.append(label)

                if self.model == "bert-bytoken":
                    new_sentences, new_tw_positions, new_labels, new_token_type_ids, max_numb_indices = position_aware_tokenization(
                        sentence_pairs, tw_positions, self.labels, tokenizer, maxlen)
                    if comparison == "noun_adjnoun":
                        tw_positions_noun_to_combine = []
                        for bertindices, noun_position in zip(new_tw_positions, secondnoun_and_position):
                            noun, position = noun_position
                            tokenized_noun = tokenizer.tokenize(noun)
                            last_adj_bert_index = bertindices['position2'][-1]
                            noun_bert_indices = []
                            i = 1
                            for _ in range(len(tokenized_noun)):
                                noun_bert_indices.append(last_adj_bert_index+i)
                                i+=1
                            tw_positions_noun_to_combine.append(noun_bert_indices)

                        max_noun_numb_indices = max([len(x) for x in tw_positions_noun_to_combine])
                        self.twnoun_tensor = torch.tensor(pad_sequences(tw_positions_noun_to_combine, maxlen=max_noun_numb_indices, padding="post"))

                    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in new_sentences],maxlen=maxlen, dtype="long", truncating="post", padding="post")
                    token_type_ids = pad_sequences(new_token_type_ids, maxlen=maxlen, dtype="long", truncating="post",padding="post", value=0)
                    tw1l = [tw["position1"] for tw in new_tw_positions]
                    tw2l = [tw["position2"] for tw in new_tw_positions]
                    tw1_tensor = torch.tensor(pad_sequences(tw1l, maxlen=max_numb_indices, padding="post"))
                    tw2_tensor = torch.tensor(pad_sequences(tw2l, maxlen=max_numb_indices, padding="post"))
                    self.tokenized_positions = torch.stack([tw1_tensor, tw2_tensor], 1)

                elif self.model == "bert":
                    input_ids = pad_sequences(input_ids, maxlen=maxlen, dtype="long", truncating="post", padding="post")
                    token_type_ids = pad_sequences(token_type_ids, maxlen=maxlen, dtype="long", truncating="post",padding="post",value=0)

                attention_masks = [[float(i != 0) for i in ii] for ii in input_ids]

                self.input_tensor = torch.tensor(input_ids)
                self.token_type_ids_tensor = torch.tensor(token_type_ids)
                self.attentionmask_tensor = torch.tensor(attention_masks)
                self.label_tensor = torch.tensor(self.labels)


        def __getitem__(self, index):
                sentence_x = self.input_tensor[index]
                label_y = self.label_tensor[index]
                attention_mask = self.attentionmask_tensor[index]               
                token_type_id = self.token_type_ids_tensor[index]
                if self.model != "bert-bytoken":
                    return sentence_x, label_y, token_type_id, attention_mask
                else:
                    position = self.tokenized_positions[index]
                    return sentence_x, label_y, token_type_id, attention_mask, position


        def __len__(self):
                return len(self.input_tensor)


def train(args, train_dataset, model, dev_dataset):

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        writer.write("EVALUATION\n")

    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.evaluate_during_training:
        dev_sampler = SequentialSampler(dev_dataset) if args.local_rank == -1 else DistributedSampler(dev_dataset)
        dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.train_batch_size)
   
    t_total = len(train_dataloader) // args.num_train_epochs
    args.n_gpu = torch.cuda.device_count()

    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              output_device=args.local_rank,
                                                              find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    for epoch in train_iterator:
        epoch_iterator= tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        ep_loss = 0.0
        for step, batch in enumerate(epoch_iterator): #
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0]}
            inputs["token_type_ids"] = batch[2]
            inputs["labels"] = batch[1]
            inputs["attention_mask"] = batch[3]
            if args.model_type == "bert-bytoken":
                inputs["input_target_indices"] = batch[4]

            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            ep_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule

                model.zero_grad()
                global_step +=1

        if args.save_epochs:
            checkpoint_prefix = 'epoch'
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, epoch))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model,
                                                'module') else model
            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logging.info("Saving model checkpoint to %s", output_dir)

        if args.evaluate_during_training:
            lossdev = 0.0
            preds = None
            out_label_ids = None
            model.eval()
            for stepdev, batch in enumerate(dev_dataloader):
                with torch.no_grad():
                    batch = tuple(t.to(args.device) for t in batch)
                    inputs = {'input_ids': batch[0]}
                    inputs["token_type_ids"] = batch[2]
                    inputs["labels"] = batch[1]
                    inputs["attention_mask"] = batch[3]
                    if args.model_type == "bert-bytoken":
                        inputs["input_target_indices"] = batch[4]

                    outputsdev = model(**inputs)
                    lossdevstep, logits = outputsdev[:2]
                    lossdev += lossdevstep.mean().item()

                    preds, out_label_ids = gather_predictions(logits, inputs, preds, out_label_ids)

            preds = np.argmax(preds, axis=1)

            one_result = evaluate_predictions(preds, out_label_ids)
            logger.info("***** Eval results on dev at epoch {} *****".format(epoch))

            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            with open(output_eval_file, "a+") as writer:
                writer.write("***** Eval results on dev at epoch {} *****\n".format(epoch))
                for key  in sorted(one_result.keys()):
                    logger.info("  %s = %s", key, str(one_result[key]))
                    writer.write("%s = %s\n" % (key, str(one_result[key])))

                if np.all(preds==1) or np.all(preds==0):               
                    logger.info("\n++++++++++ All predictions are the same! +++++++")
                    writer.write("++++++++++ All predictions are the same! +++++++")

                else:
                    logger.info("\n********** Not all predictions are the same *******")
                    writer.write("********** Not all predictions are the same *******")

                if list(preds).count(1) > len(preds)*0.9 or list(preds).count(0) > len(preds)*0.9:
                    logger.info("\n~~~~~~~~~~~ One prediction is a big majority ~~~~~~~~~~~")
                    writer.write("~~~~~~~~~~~ One prediction is a big majority ~~~~~~~~~~~")

                writer.write("\nAv Epoch Train loss =" + str(ep_loss / (step+1)) + "\n")

            if epoch == args.num_train_epochs-1:
                with open(os.path.join(args.output_dir, "last_dev_f1.txt"), "w") as out:
                    out.write(str(one_result['f1']))

        logger.info("Av Epoch Train loss = %f", ep_loss / (step+1))
        if args.evaluate_during_training:
            logger.info("Av Epoch Dev loss = %f", lossdev / (stepdev+1))
            if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)
            torch.save([("epoch", epoch), ("training_loss", ep_loss / (step+1)), ("dev loss", lossdev / (stepdev+1))], os.path.join(args.output_dir, 'losses.bin'))

    return global_step, tr_loss / global_step

def gather_predictions(logits, inputs, preds, out_label_ids):
    if preds is None:
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs['labels'].detach().cpu().numpy()
    else:
        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    return preds, out_label_ids


def evaluate_predictions(preds, out_label_ids):
    oneresult = {}
    oneresult["accuracy"] = accuracy_score(out_label_ids, preds)
    oneresult["precision"] = precision_score(out_label_ids, preds, pos_label=1)
    oneresult["recall"] = recall_score(out_label_ids, preds, pos_label=1)
    oneresult["f1"] = f1_score(out_label_ids, preds, pos_label=1)
    return oneresult


def evaluate(args, model, prefix=""):
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(args.eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(args.eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(args.eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0]}
            inputs["token_type_ids"] = batch[2]
            inputs["labels"] = batch[1]
            inputs["attention_mask"] = batch[3]
            if args.model_type == "bert-bytoken":
                inputs["input_target_indices"] = batch[4]

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            preds, out_label_ids = gather_predictions(logits, inputs, preds, out_label_ids)
    eval_loss = eval_loss / nb_eval_steps
    print("EVALUATION LOSS", eval_loss)

    if args.save_preds:
        pickle.dump(preds, open(args.output_dir + "output_probs.pkl", "wb"))

    preds = np.argmax(preds, axis=1)

    one_result = evaluate_predictions(preds, out_label_ids)

    if np.all(preds==1) or np.all(preds==0):
        logger.info("\n++++++++++ All predictions are the same +++++++")
    else:
        logger.info("\n********** Not all predictions are the same! *******")

    if args.save_preds:
        pickle.dump(preds, open(args.output_dir + "predictions.pkl", "wb"))

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "a+") as writer:
        writer.write("***** Final Eval results {} *****\n".format(prefix))
        logger.info("***** Eval results {} *****".format(prefix))
        for key  in sorted(one_result.keys()):
            logger.info("  %s = %s", key, str(one_result[key]))
            writer.write("%s = %s\n" % (key, str(one_result[key])))

    return one_result


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--model_type", type=str, required=True,
                        help="bert (for bert (cls)) or bert-bytoken (for bert (tok))")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name. We used bert-base-uncased")
    parser.add_argument('--dataset', required=True, help="Dataset on which to train the model. It can be addone or HVD.")
    # Other
    parser.add_argument('--comparison', help="For bert-bytoken, the words that are compared: it can be noun_noun (for N_sN, N_sAN),"
                            "noun_adj (N_sN, A_sAN), and noun_adjnoun (for N_sN, N_sAN+A_sAN)")
    parser.add_argument('--test_fold', default=1, type=int, help="What fold is used for testing (for HVD).")
    parser.add_argument("--output_dir", default="FT_results/", type=str, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length", default=512, type=int, help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=40,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", default=False, action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", default=True, action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--fp16_opt_level', type=str, default='O1', help = "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."         "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="dropout probability")
    parser.add_argument('--save_preds', action='store_true', help="Save a file with predictions")
    parser.add_argument('--save_epochs', action='store_true', help="Whether to save every epoch")
    parser.add_argument('--eval_on_dev', action='store_true', help="Whether to evaluate on the development set instead of the test set")
    parser.add_argument('--eval_dataset', type=str, help = "The dataset on which we perform evaluation (adddone or HVD)")
    parser.add_argument('--save_model', action='store_true', help="whether to save the model after training.")

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    if args.dataset == "HVD" and (not args.test_fold or not args.comparison):
        sys.exit("Indicate the test fold and/or the comparison to run (see argument options).")

    # Setup logging
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)        
    logging.basicConfig(filename=args.output_dir + "log", format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model_class = MODEL_CLASSES[args.model_type]
    config_class = BertConfig
    num_labels = 2
    config = config_class.from_pretrained(args.model_name_or_path, num_labels=num_labels)

    config.hidden_dropout_prob = args.dropout_prob
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True)

    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.dataset == "addone":
            train_dataset = AddOneDataset("train", tokenizer, model=args.model_type, maxlen=args.max_seq_length, comparison=args.comparison)
        elif args.dataset == "HVD":
             train_dataset = McRaeDataset(args.dataset, "train", tokenizer, model=args.model_type, maxlen=args.max_seq_length, comparison=args.comparison, test_fold=args.test_fold)
        
        dev_dataset = []
        if args.evaluate_during_training:
            if args.dataset == "addone":
                dev_dataset = AddOneDataset("dev", tokenizer, model=args.model_type,  maxlen=args.max_seq_length, comparison=args.comparison)
            elif args.dataset == "HVD":
                dev_dataset = McRaeDataset(args.dataset, "dev", tokenizer, model=args.model_type, maxlen=args.max_seq_length, comparison=args.comparison, test_fold=args.test_fold)

        global_step, tr_loss = train(args, train_dataset, model, dev_dataset)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        if args.save_model:
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(args.output_dir)

        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        if args.save_model:
            model = model_class.from_pretrained(args.output_dir)

        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True)

        model.to(device)
    result = []

    if args.do_eval:
        if args.eval_on_dev:
            subset_to_eval = "dev"
        else:
            subset_to_eval = "test"
        if args.eval_dataset == "addone":
            args.eval_dataset = AddOneDataset(subset_to_eval, tokenizer, model=args.model_type, maxlen=args.max_seq_length, comparison=args.comparison)
        elif args.eval_dataset == "HVD":
            args.eval_dataset = McRaeDataset(args.eval_dataset, subset_to_eval, tokenizer, model=args.model_type, maxlen=args.max_seq_length, comparison=args.comparison, test_fold=args.test_fold)

    if args.do_train and args.do_eval and args.local_rank in [-1, 0]:

        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in
                               sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        result = evaluate(args, model)

    elif args.do_eval and not args.do_train:
        results = dict()
        model.to(args.device)
        result = evaluate(args, model)
        result = dict((k, v) for k, v in result.items())
        results.update(result)

    return result

if __name__ == '__main__':
    main()


