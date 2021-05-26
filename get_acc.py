import json
import os,sys
import numpy as np

pred_file = sys.argv[1]
def read_preds(file, ismale = 0):
    preds = json.load(open(file, 'r'))
    golden = []
    ans_q0 = []
    ans_q1 = []
    occs = []
    for key in preds:
        tks = key.strip().split('|')
        if ismale == 1:
            if tks[0] == 'male':
                correct_ans = tks[2]
            else:
                correct_ans = tks[3]
        else:
            if tks[0] == 'male':
                correct_ans = tks[3]
            else:
                correct_ans = tks[2]
                
        golden.append(correct_ans)
        ans_q0.append(preds[key]['q0']['pred'])
        ans_q1.append(preds[key]['q1']['pred'])
        occs.append(tks[6].split()[-1])
    return golden, ans_q0, ans_q1, occs

def cal_same(golden, ans_q0):
    count = 0
    assert len(golden) == len(ans_q0)
    for idx in range(len(golden)):
        if golden[idx] in ''.join(ans_q0[idx].split()):
            count += 1
    print(count/len(ans_q0))
    return count/len(ans_q0)
        
if __name__ == '__main__':
    if 'extraMale' in pred_file:
        ismale = 1
    else:
        ismale = 0
    golden, ansq0, _,_ = read_preds(pred_file, ismale)
    acc = cal_same(golden, ansq0)
    print(f"{pred_file} acc: {acc}")