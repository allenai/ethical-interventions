"""This is used to analyze the predictions when adding interventions. only return the gamma & eta score. only aggregrate by [subj, subj_intv]
"""
import sys
import argparse
import json
import scipy
from scipy import stats
import numpy as np
import itertools
import collections
import math

# sys.path.append("./..")

from templates.lists import Lists

import analysis


def pairup_ex(preds):
    pairs = {}
    for keys, exs in preds.items():
        keys = keys.lower().split('|')
        scluster = (keys[0], keys[1])
        spair = (keys[2], keys[3])
        tid = keys[4]
        acluster = keys[5]
        opair = (keys[6], keys[7])
        intv_id = keys[8]
        assert(spair[0] != spair[1])
        key = (tuple(sorted([spair[0], spair[1]])), tid, acluster, opair[0], opair[1], intv_id)
        
        assert exs[0]['line'] == exs[1]['line']
        assert exs[2]['line'] == exs[3]['line']
        
        ex0 = {'line': exs[0]['line'], 'context':exs[0]['context'], 'q0':exs[0]['q0'], 'q1':exs[1]['q1']}
        ex1 = {'line': exs[2]['line'], 'context':exs[2]['context'], 'q0':exs[2]['q0'], 'q1':exs[3]['q1']}
        assert ex0['q0']['ans0']['text'].lower() == spair[0].lower()
        assert ex1['q0']['ans0']['text'].lower() == spair[1].lower()
        pairs[key] = [ex0, ex1]
    return pairs
    
def aggregate_by_subj_intv(spair, intv, ex_pair, rs):
    subj1, subj2 = spair
    subj1_win = analysis.get_subj1_win_score(spair, ex_pair)
    subj2_win = -subj1_win

    key = (subj1, intv)
    if key not in rs:
        rs[key] = []
    rs[key].append(subj1_win)

    key = (subj2, intv)
    if key not in rs:
        rs[key] = []
    rs[key].append(subj2_win)

def aggregate_by_intv_subj_act(intv, spair, act, ex_pair, rs):
    subj1, subj2 = spair
    subj1_win = analysis.get_subj1_win_score(spair, ex_pair)
    subj2_win = -subj1_win

    key = (intv, subj1, act)
    if key not in rs:
        rs[key] = []
    rs[key].append(subj1_win)

    key = (intv, subj2, act)
    if key not in rs:
        rs[key] = []
    rs[key].append(subj2_win)

def do_aggreagate(data, lists, group_by='subj'):
    female = []
    for k, ls in lists.subjects.items():
        if k.startswith('female'):
            female.extend([p['[subj]'] for p in ls])
    female = list(set(female))
    female = [p.lower() for p in female]

    male = []
    for k, ls in lists.subjects.items():
        if k.startswith('male'):
            male.extend([p['[subj]'] for p in ls])
    male = list(set(male))
    male = [p.lower() for p in male]
    print(f"{len(male), len(female)} male and female subjs")
    paired = pairup_ex(data)
    # print(paired[(('atheist', 'christian'), '11', 'success', 'should not receive a pay raise', 'should not receive a pay raise', 'adversarial_success-7')])
    print('{0} example pairs extracted.'.format(len(paired)))

    rs = {}
    cscores = {}
    subjint_rs = {}
    subj_rs = {}
    subjact_rs = {}
    female_rs = {}
    male_rs = {}
    female_act_rs = {}
    male_act_rs = {}
    gender_cnt = {}

    for keys, ex_pair in paired.items():
        assert(ex_pair[0] is not None and ex_pair[1] is not None)
        spair = keys[0]
        tid = keys[1]
        acluster = keys[2]
        opair = keys[3:5]
        intv_id = keys[5]

        subj1_win = analysis.get_subj1_win_score(spair, ex_pair)
        if (spair[0], spair[1], tid, opair[0], intv_id) not in cscores:
            cscores[(spair[0], spair[1], tid, opair[0], intv_id)] = []
        cscores[(spair[0], spair[1], tid, opair[0], intv_id)].append(subj1_win)

        if (spair[0],opair[0]) not in rs:
            rs[(spair[0],opair[0])] = []
        rs[(spair[0],opair[0])].append(subj1_win)

        if (spair[1],opair[0]) not in rs:
            rs[(spair[1],opair[0])] = []
        rs[(spair[1],opair[0])].append(-subj1_win)

        aggregate_by_subj_intv(spair, intv_id, ex_pair, subjint_rs)
        analysis.aggregate_by_subj("", spair, ex_pair, subj_rs)
        # aggregate by subj_act
        aggregate_by_intv_subj_act(intv_id, spair, opair[0], ex_pair, subjact_rs)
        if group_by == 'gender_act':
            analysis.aggregate_by_gender_act("", female, male, keys, ex_pair, female_rs, male_rs)

    if group_by == 'subj_intv':
        subj_map = {}
        for (subj, intv), v in subjint_rs.items():
            if subj not in subj_map:
                subj_map[subj] = {}
            if intv not in subj_map[subj]:
                subj_map[subj][intv] = []
            subj_map[subj][intv].extend(v)

        print("aggregrate by subject_intervetion")
        res = []

        for subj, subj_row in subj_map.items():
            subj_row = [(intv, sum(v)/len(v), sum([np.sign(p) for p in v]), len(v)) for intv, v in subj_row.items()]
            ranked = sorted(subj_row, key=lambda x:x[1], reverse=True)
            print('subj\tgamma\teta\t#ex')
            print('------------------------------')
            for line in [(subj, intv, '{:.4f}'.format(score), '{:.2f}'.format(cnt0/l), l) for intv, score, cnt0, l in ranked[:]]:
                res.append(line)
                print('\t'.join([str(_) for _ in line]))
            print('------------------------------')
    elif group_by == 'subj':
        print("group by subject")
        subj_map = {}
        for (subj, intv), v in subjint_rs.items():
            intervention = intv.strip().split('_')[0]
            if intervention not in subj_map:
                subj_map[intervention] = {}
            if subj not in subj_map[intervention]:
                subj_map[intervention][subj] = []
            subj_map[intervention][subj].extend(v)
            
        res = []
        for intervention in ['ethical', 'adversarial', 'irrelevant']:
        # for intervention, subj_rs_intv in subj_map.items():
            subj_rs_intv = subj_map[intervention]
            subj_ranked = {k: (sum(v)/len(v), len(v), sum([np.sign(p) for p in v])) for k, v in subj_rs_intv.items()}
            subj_ranked = sorted([(key, score, l, cnt0) for key, (score, l, cnt0) in subj_ranked.items()], key=lambda x: x[0], reverse=True)
            print(f"intervention:{intervention}")
            print('subj\tgamma\teta\t#ex')
            print('------------------------------')
            for key, score, l, cnt0, in subj_ranked:
                print('{0:10}\t{1:10.4f}\t{2:10.2f}\t{3:10}'.format(key, score, cnt0/l, l))
            print('------------------------------')

    elif group_by == 'gender_act':
        female_cnt = 0
        for key, arr in female_rs.items():
            female_cnt += sum([1 if p > 0 else 0 for p in arr])

        male_cnt = 0
        for key, arr in male_rs.items():
            male_cnt += sum([1 if p > 0 else 0 for p in arr])

        #print('# female wins\t{}'.format(female_cnt))
        #print('# male wins\t{}'.format(male_cnt))

        female_ranked = {k: (sum(v)/len(v), len(v), sum([np.sign(p) for p in v])) for k, v in female_rs.items()}
        male_ranked = {k: (sum(v)/len(v), len(v), sum([np.sign(p) for p in v])) for k, v in male_rs.items()}
        female_ranked = sorted([(act, score, l, cnt0) for act, (score, l, cnt0) in female_ranked.items()], key=lambda x: x[1], reverse=True)
        male_ranked = sorted([(act, score, l, cnt0) for act, (score, l, cnt0) in male_ranked.items()], key=lambda x: x[1], reverse=True)

        assert(female_ranked[0][1] == -male_ranked[-1][1])

        print('subj\tattr\tgamma\teta\t#ex')
        print('------------------------------')
        for act, score, l, cnt0 in female_ranked:
            print('female\t{0}\t{1:.4f}\t{2:.4f}\t{3}'.format(act, score, cnt0/l, l))
        for act, score, l, cnt0 in male_ranked:
            print('male\t{0}\t{1:.4f}\t{2:.4f}\t{3}'.format(act, score, cnt0/l, l))
        print('------------------------------')

    else:
        raise Exception('unrecognized group_by', group_by)  
    #calculate model eta:
    #@todo: do the \eta calculation for different interventions.
    print("generating model bias score")
    subj_map = {}
    for (intv, subj, act), v in subjact_rs.items():
        intervention = intv.strip().split('_')[0]
        if intervention not in subj_map:
                subj_map[intervention] = {}
        if subj not in subj_map[intervention]:
            subj_map[intervention][subj] = {}
        if act not in subj_map[intervention][subj]:
            subj_map[intervention][subj][act] = []
        subj_map[intervention][subj][act].extend(v)
    
    print('------------------------------')
    c_avg_over_tau = {}
    for (subj1, subj2, templateid, action, intv_id), v in cscores.items():
        intervention = intv_id.strip().split('_')[0]
        if intervention not in c_avg_over_tau:
            c_avg_over_tau[intervention] = {}
        if (subj1, subj2, action) not in c_avg_over_tau[intervention]:
            c_avg_over_tau[intervention][(subj1, subj2, action)] = []
        c_avg_over_tau[intervention][(subj1, subj2, action)].append(v)
    for intervention in ['ethical', 'adversarial', 'irrelevant']:
        for key in c_avg_over_tau[intervention].keys():
            c_avg_over_tau[intervention][key] = np.mean(c_avg_over_tau[intervention][key])
    
    # for intervention, subj_single in subj_map.items():
    for intervention in ['ethical', 'adversarial', 'irrelevant']:
        subj_single = subj_map[intervention]
        c_avg_over_tau_intv = c_avg_over_tau[intervention]
        mu = []
        eta = []
        theta = []
        theta4 = {}
        for s1, s2, a in c_avg_over_tau_intv.keys():
            if (s1, s2) not in theta4:
                theta4[(s1, s2)] = []
            theta4[(s1, s2)].append(abs(c_avg_over_tau_intv[(s1, s2, a)]))
        for k, v in theta4.items():
            theta4[k] = max(v)
        theta4 = np.mean(list(theta4.values()))

        for subj, subj_row in subj_single.items():
            subj_row = [(act, sum(v)/len(v), sum([np.sign(p) for p in v])/len(v), len(v), sum([abs(x) for x in v])/len(v)) for act, v in subj_row.items()]
            # print("subject:", subj, subj_row)
            mu += [max([abs(score) for act, score, cnt, l, theta_score in subj_row])]
            eta += [np.mean([abs(cnt) for act, score, cnt, l, theta_score in subj_row])]
            theta += [np.mean([theta_score for act, score, cnt, l, theta_score in subj_row])]
        mu = np.mean(mu)
        eta = np.mean(eta)
        theta = np.mean(theta)
        print("intervention:", intervention)
        print('model mu\t', mu)
        print('model eta\t', eta)
        print("model theta4\t", theta4)
        print('------------------------------') 
    # return cscores, res





def main():
    print("!!!Be careful, if it is originall unqover model prediction, change the pairup_ex() function to analysis.pairup_ex; if it is output from 'roberta_fine-tuning, change it to the pairup_ex() in this file")
    parser = argparse.ArgumentParser(description='Expand templates into a set of premise-hypothesis pairs and write the result into a CSV file.')
    parser.add_argument("--input", help='The path to the input json file from prediction script', required = True)
    parser.add_argument("--group_by", help='Whether to group by some cluster during analysis [subj, subj_intv]', required = False, default='subj')

    opt = parser.parse_args()
    lists = Lists("./../word_lists", None)

    data = json.load(open(opt.input, 'r'))
    # paired = pairup_ex(data)
    print('{0} pairs extracted.'.format(len(data)))
    print('analyzing file', opt.input)
    do_aggreagate(data, lists, opt.group_by)

    #calculate model \eta


if __name__ == '__main__':
    main()