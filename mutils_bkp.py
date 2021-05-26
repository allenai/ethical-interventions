import json
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_roberta import RobertaForQuestionAnswering 
from transformers import RobertaTokenizerFast


def read_squad(path, count=-1):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    
    if count != -1:
        return contexts[:count], questions[:count], answers[:count]

    return contexts, questions, answers

def read_data(path, count = -1):
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    interventions = []
    ids = []
    keys = []
    for group in squad_dict["data"]:
        for passage in group['paragraphs']:
            context = passage['context']
            intervention = passage['intervention']
            # if intervention == 'irrelevant':
            #     continue
            for qa in passage['qas']:
                question = qa['question']
                contexts.append(context)
                questions.append(question)
                interventions.append(intervention)
                # for answer in qa['answers']:
                answers.append(sorted(qa['answers'], key = lambda i: i['start-end-score'][-1], reverse=True)) #sort the answers by the original prepdiction score. [higher-score; lower-score]
                ids.append(qa['id']) #line+questionid 
                keys.append(passage['key'])
                if intervention != passage['key'].split('|')[-1].split('_')[0]:
                    print("wrong intervention or key; check the data")
                    exit()

    assert len(contexts) == len(questions) == len(answers) == len(interventions) == len(ids) == len(keys)
    if count != -1:
        contexts, questions, answers, interventions, ids, keys = contexts[:count], questions[:count], answers[:count], interventions[:count], ids[:count], keys[:count]
#     entities = defaultdict(list)
#     for keyid in range(len(keys)):
#         if interventions[keyid] == 'irrelevant':
#             continue
#         entities[keys[keyid].strip().split('|')[0]].append(keys[keyid].strip().split('|')[-1])
#         entities[keys[keyid].strip().split('|')[1]].append(keys[keyid].strip().split('|')[-1])
#     for k in entities:
#         print(f'{k}: {Counter(entities[k])}')

    return contexts, questions, answers, interventions, ids, keys


def add_end_idx(answers, contexts):
    for answertuple, context in zip(answers, contexts):
        for answer in answertuple:
            gold_text = answer['text']
            start_idx = answer['answer_start']
            end_idx = start_idx + len(gold_text)

            # sometimes squad answers are off by a character or two – fix this
            if context[start_idx:end_idx] == gold_text:
                answer['answer_end'] = end_idx
            elif context[start_idx-1:end_idx-1] == gold_text:
                answer['answer_start'] = start_idx - 1
                answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
            elif context[start_idx-2:end_idx-2] == gold_text:
                answer['answer_start'] = start_idx - 2
                answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters


#convert our character start/end positions to token start/end positions.
def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    texts = []
    for i in range(len(answers)):
        tmp_start = []
        tmp_end = []
        text = []
        for j in range(len(answers[i])):
            tmp_start.append(encodings.char_to_token(i, answers[i][j]['answer_start']))
            tmp_end.append(encodings.char_to_token(i, answers[i][j]['answer_end'] - 1))
            # if None, the answer passage has been truncated
            if tmp_start[-1] is None:
                tmp_start[-1] = tokenizer.model_max_length
            if tmp_end[-1] is None:
                tmp_end[-1] = tokenizer.model_max_length
            text.append(answers[i][j]['text'])
        start_positions.append(tmp_start)
        end_positions.append(tmp_end)
        texts.append(text)
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
    print("update the start/end positions")



def add_interventions(encodings, interventions):
    intervention2id = {'ethical':0, 'adversarial': 1, 'irrelevant': 2}
    interventionids = []
    for i in range(len(interventions)):
        intervention = interventions[i]
        interventionids.append(intervention2id[intervention])
    encodings.update({'interventions': interventionids})
    print("added intervetions to the data")

def get_tokens(encodings, tokenizer):
    tokens = []
    for i in range(len(encodings["input_ids"])):
        token = tokenizer.convert_ids_to_tokens(encodings['input_ids'][i])
        tokens.append(token)
    return tokens




class SquadFormat(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def get_data(filename, tokenizer, count=-1):
    contexts, questions, answers, interventions, ids, keys = read_data(filename, count)
    assert len(contexts) == len(questions) == len(answers) == len(interventions) == len(ids) == len(keys) 
    print(f"loaded {len(contexts)} examples from {filename}") 
    print(f"{len([x for x in interventions if x=='ethical'])} ethical interventions, {len([x for x in interventions if x=='adversarial'])} adversarial interventions; {len([x for x in interventions if x=='irrelevant'])} irrelevant")
    add_end_idx(answers, contexts)
    encodings = tokenizer(contexts, questions, truncation=True, padding=True)
    add_token_positions(encodings, answers)
    add_interventions(encodings, interventions)
    dataset = SquadFormat(encodings)
    tokens = get_tokens(encodings, tokenizer)
    # print("dataset:", dataset[:4])
    # print("tokens:", tokens[:4])
    # print("ids:", ids[:4])
    # print("keys:", keys[:4])
    # print("contexts:", contexts[:4])
    # print("questions:", questions[:4])
    # print("answers:", answers[:4])
    return dataset, tokens, ids, keys, contexts, questions, answers

def get_squad_data(path, tokenizer, count = -1):
    contexts, questions, answers = read_squad(path, count=count)
    print(f"load {len(contexts)} from {path}")
    add_end_idx(answers, contexts)
    encodings = tokenizer(contexts, questions, truncation = True, padding = True)
    add_token_positions(encodings, answers)
    dataset = SquadFormat(encodings)
    return dataset


class RobertaForQuestionAnsweringonUnqover(RobertaForQuestionAnswering):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        interventions = None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        # print("Inside model:")
        # print("start logits:", start_logits[:2])
        # print("interventions:", interventions)
        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            ####### change loss functions wrt intervention types. 0-ethical interventions; 1-adversarial; 2-irrelevant
            ethical_ids = torch.where(interventions == 0) #calculate the loss w.r.t 2 subjects
            adversarial_ids = torch.where(interventions == 1) #calcuate the loss w.r.t. 1st subject answer (make the gap larger)
            irrelevant_ids = torch.where(interventions == 2)
            
            assert len(ethical_ids[0]) + len(adversarial_ids[0]) + len(irrelevant_ids[0]) == input_ids.size()[0]
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            coef0 = 1
            coef1 = 0
            start_softmax = torch.nn.functional.softmax(start_logits, dim=-1)
            end_softmax = torch.nn.functional.softmax(end_logits, dim=-1)

            # print("start positions:", start_positions)
            if len(adversarial_ids[0]) > 0:
                # start_loss_adv = loss_fct(start_logits[adversarial_ids], start_positions[adversarial_ids][:, 0])
                # end_loss_adv = loss_fct(end_logits[adversarial_ids], end_positions[adversarial_ids][:, 0])
                # total_loss = (start_loss_adv + end_loss_adv)

                start_logit_adv_subj0 = torch.gather(input = start_softmax[adversarial_ids], dim = -1, index = start_positions[adversarial_ids][:, 0].unsqueeze(-1))
                # print("start logit subj0:", start_logit_adv_subj0)
                start_logit_adv_subj1 = torch.gather(input = start_softmax[adversarial_ids], dim = -1, index = start_positions[adversarial_ids][:, 1].unsqueeze(-1))
                # print("start logit subj1:", start_logit_adv_subj1)
                end_logit_adv_subj0 = torch.gather(input = end_softmax[adversarial_ids], dim = -1, index = end_positions[adversarial_ids][:, 0].unsqueeze(-1))
                end_logit_adv_subj1 = torch.gather(input = end_softmax[adversarial_ids], dim = -1, index = end_positions[adversarial_ids][:, 1].unsqueeze(-1))
                assert(start_logit_adv_subj0.size() == end_logit_adv_subj0.size() == (len(adversarial_ids[0]), 1))
                total_loss = - (torch.abs(start_logit_adv_subj0  - start_logit_adv_subj1 + end_logit_adv_subj0 - end_logit_adv_subj1).sum())*coef0 \
                    - (start_logit_adv_subj0 + end_logit_adv_subj0 + start_logit_adv_subj1 + end_logit_adv_subj1).sum() * coef1
            if len(ethical_ids[0]) > 0:   
                '''w/ ethical intervetions, make the gap between to entities as small as possible'''
                # start_loss_ethical_0 = loss_fct(start_logits[ethical_ids], start_positions[ethical_ids][:, 0]) 
                # start_loss_ethical_1 = loss_fct(start_logits[ethical_ids], start_positions[ethical_ids][:, 1])
                # end_loss_ethical_0 = loss_fct(end_logits[ethical_ids], end_positions[ethical_ids][:, 0]) 
                # end_loss_ethical_1 = loss_fct(end_logits[ethical_ids], end_positions[ethical_ids][:, 1])
                start_logit_ethical_subj0 = torch.gather(input = start_softmax[ethical_ids], dim = -1, index = start_positions[ethical_ids][:, 0].unsqueeze(-1))

                start_logit_ethical_subj1 = torch.gather(input = start_softmax[ethical_ids], dim = -1, index = start_positions[ethical_ids][:, 1].unsqueeze(-1))
                end_logit_ethical_subj0 = torch.gather(input = end_softmax[ethical_ids], dim = -1, index = end_positions[ethical_ids][:, 0].unsqueeze(-1))
                end_logit_ethical_subj1 = torch.gather(input = end_softmax[ethical_ids], dim = -1, index = end_positions[ethical_ids][:, 1].unsqueeze(-1))
                
                if total_loss != None:
                    # total_loss += (start_loss_ethical_0 + end_loss_ethical_0 + start_loss_ethical_1 + end_loss_ethical_1) / 2
                    total_loss += ((torch.abs(start_logit_ethical_subj0 - start_logit_ethical_subj1 \
                    + end_logit_ethical_subj0 - end_logit_ethical_subj1).sum()) * coef0 \
                        - (start_logit_ethical_subj0 + end_logit_ethical_subj0 + start_logit_ethical_subj1 + end_logit_ethical_subj1).sum() * coef1) 
                else:
                    total_loss = ((torch.abs(start_logit_ethical_subj0 - start_logit_ethical_subj1 \
                    + end_logit_ethical_subj0 - end_logit_ethical_subj1).sum()) * coef0 \
                        - (start_logit_ethical_subj0 + end_logit_ethical_subj0 + start_logit_ethical_subj1 + end_logit_ethical_subj1).sum() * coef1)


            if len(irrelevant_ids[0]) > 0: 
                ''' may need to change the loss function for irrelevant internvetions later.
                '''
                raise ValueError("have irrelevant interventions in the batch")
                exit()
                # start_loss_irrelevant = loss_fct(start_logits[irrelevant_ids], start_positions[irrelevant_ids][:, 0])
                # end_loss_irrelevant = loss_fct(end_logits[irrelevant_ids], end_positions[irrelevant_ids][:, 0])
                start_logit_irrelevant_subj0 = start_logits[irrelevant_ids][:, start_positions[irrelevant_ids][:, 0]]
                start_logit_irrelevant_subj1 = start_logits[irrelevant_ids][:, start_positions[irrelevant_ids][:, 1]]
                end_logit_irrelevant_subj0 = end_logits[irrelevant_ids][:, end_positions[irrelevant_ids][:, 0]]
                end_logit_irrelevant_subj1 = end_logits[irrelevant_ids][:, end_positions[irrelevant_ids][:, 1]]
                if total_loss != None:
                    total_loss += (start_logit_irrelevant_subj0 + start_logit_irrelevant_subj1 + end_logit_irrelevant_subj0 + end_logit_irrelevant_subj1)
                else:
                    total_loss =  (start_logit_irrelevant_subj0 + start_logit_irrelevant_subj1 + end_logit_irrelevant_subj0 + end_logit_irrelevant_subj1)
            
            total_loss = total_loss / 2
            # print("Loss:", total_loss)
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
            



    

def main():
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    model = RobertaForQuestionAnsweringonUnqover.from_pretrained('roberta-base')

    from torch.utils.data import DataLoader
    from transformers import AdamW

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    model.train()

    # train_dataset = get_data("./../data/train_squad.json", tokenizer)
    dataset, all_tokens, all_ids, all_keys, all_contexts, all_questions, all_answers = get_data(os.path.join('./../data/religion/noIntvOverlap', 'dev_squad.json'), tokenizer, count=10000)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)
    print(list(iter(dev_loader))[0])
    exit()

    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(100):
        total_loss = 0.
        for batch in dev_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            interventions = batch['interventions'].to(device)
            non_irrelevant = torch.where(interventions != 2)
            if len(non_irrelevant[0]) < 1: #now skip the irrelevant examples
                continue
            outputs = model(input_ids[non_irrelevant], attention_mask=attention_mask[non_irrelevant], start_positions=start_positions[non_irrelevant], end_positions=end_positions[non_irrelevant], interventions=interventions[non_irrelevant])
            loss = outputs[0]
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"{epoch}-th epoch, loss:{total_loss}")
    model.eval()



if __name__ == '__main__':
    main()