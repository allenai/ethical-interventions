"""
adding the c_mean_over_tau to the adversarial training examples
"""
import json
import sys, os

num=sys.argv[1]
training = json.load(open(f'/home/jieyuz/git/unqover/data/religion/noniids/train_{num}.json'))
c_mean_over_tau = json.load(open('/home/jieyuz/git/unqover/data/religion/zeroshotEval/robertalarge_religion-adversarial-bias.json'))
newtrain = {}
print(len(training)*4)
for key, ex in training.items():
    tks = key.split('|')
    ckey = '|'.join([tks[0], tks[1], tks[4]])
    interventions = tks[-1].split('_')[0]
    if interventions == 'adversarial':
        if ckey not in c_mean_over_tau:
            print("wrong example")
            exit()
        if c_mean_over_tau[ckey] > 0: #bias(x1,x2,a) > 0 --> for q0: maximize score for x1; otherwise maximize score for x2
            ex[0]['q0']['max'] = 0 #non-negation question; x1 is in the first position
            ex[0]['q1']['max'] = 1 #negation question; x1 is in the 1st position
            ex[1]['q0']['max'] = 1 #non-negation question; x1 is in the 2nd position
            ex[1]['q1']['max'] = 0 #negation question; x1 is in the 2nd position
        else:
            ex[0]['q0']['max'] = 1 #non-negation question; x1 is in the 1st position
            ex[0]['q1']['max'] = 0 #negation question; x1 is in the 1st position
            ex[1]['q0']['max'] = 0 #non-negation question; x1 is in the 2nd position
            ex[1]['q1']['max'] = 1 #negation question; x1 is in the 2nd position
    else: #for other interventions, just add placeholder
        ex[0]['q0']['max'] = -1
        ex[0]['q1']['max'] = -1
        ex[1]['q0']['max'] = -1
        ex[1]['q1']['max'] = -1
    newtrain[key] = ex
json.dump(newtrain, open(f"/home/jieyuz/git/unqover/data/religion/noniids/addedbias_train_{num}.json", 'w'))
    
