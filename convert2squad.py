import json
import numpy as np
import math
import os,sys

# datapath = './../data/noIntvOverlap'
datapath = sys.argv[1]
num = sys.argv[2]
# for data in [ 'dev', 'test', 'train']:
for data in ['train']:
    dev = json.load(open(os.path.join(datapath, f'addedbias_train_{num}.json'), 'r')) 

    squad = {'version': "squad, iid", "data":[]}
    count = 0
    for key, queans in dev.items():
        
        for qs in queans[:2]: #different context; remove the avgscore in the religion case
            tmp = {"qas":[]}
            tmp["key"] = key
            tmp["context"] = qs["context"]
            intervention = key.strip().split("|")[-1].split('_')[0]
            for q in ["q0", "q1"]:
                question0 = qs[q]['question']
                id0 = str(qs["line"]) +str(q[-1])
                tomax = qs[q]['max']
                answers = []
                for ansid in ['ans0', 'ans1']:
                    # print(q, qs[q])
                    ans = {'text': qs[q][ansid]["text"]}
                    ans["answer_start"] = qs['context'].find(qs['q0'][ansid]["text"])
                    ans['start-end-score'] = (qs[q][ansid]['start'], qs[q][ansid]['end'])
                    answers.append(ans)
                # if intervention == 'ethical':  #change the answer to opposite model prediction
                #     ansid = np.argmin([answers[0]['start-end-score'][-1], answers[1]['start-end-score'][-1]])
                # elif intervention in ['adversarial', 'irrelevant']: #follow the existing prediction
                #     ansid = np.argmax([answers[0]['start-end-score'][-1], answers[1]['start-end-score'][-1]])
                # else:
                #     print("wrong interventions")
                #     os._exit(0)
                tmp["qas"].append({"id": id0, "question": question0,\
                    "answers": answers, "tomax": tomax, "is_impossible": "false"})
            tmp["intervention"] = intervention
            inst = {'paragraphs': [tmp], 'title': '-'.join(key.split("|")[:2])}
            squad["data"].append(inst)


            
    json.dump(squad, open(os.path.join(datapath, f'addedbias_train_{num}_squad.json'), 'w'))