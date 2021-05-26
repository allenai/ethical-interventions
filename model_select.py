import glob
import os,sys
import numpy as np

def get_data(datapth):
    allfiles = []
    for root, dirs, files in os.walk(datapath):
        for f in files:
            if(f.endswith("dev.log.txt")):
                allfiles.append(os.path.join(root,f))
    
    checkpoints = []
    res= {}
    for infile in allfiles:
        fn=infile.split('/')[-1]
        tks = fn.strip().split('_')
        assert tks[3].split('-')[0] == 'religion'
        lr = tks[1]
        nepochs = tks[2]
        dashcount = 0
        key = lr+"_"+nepochs
        res[key] = {'adversarial': [], 'ethical': [], 'irrelevant':[]}
        with open(infile, 'r') as f:
            for line in f.readlines():
                if line.startswith('----'): #starting point
                    dashcount += 1
                    continue
                if dashcount  == 1:
                    toks = line.strip().split('\t')
                    toks = list(map(lambda x: x.strip(), toks))
                    res[key]['ethical'].append(toks)
                if dashcount == 3:
                    toks = line.strip().split('\t')
                    toks = list(map(lambda x: x.strip(), toks))
                    res[key]['adversarial'].append(toks)
                if dashcount == 5:
                    toks = line.strip().split('\t')
                    toks = list(map(lambda x: x.strip(), toks))
                    res[key]['irrelevant'].append(toks)
                if dashcount == 7:
                    if line.startswith("intervention"):
                        assert line.strip().split()[1] == 'ethical'
                        continue
                    toks = line.strip().split()
                    res[key]['ethical'].append(toks[1:])
                if dashcount == 8:
                    if line.startswith("intervention"):
                        assert line.strip().split()[1] == 'adversarial'
                        continue
                    toks = line.strip().split()
                    res[key]['adversarial'].append(toks[1:])
                if dashcount == 9:
                    if line.startswith("intervention"):
                        assert line.strip().split()[1] == 'irrelevant'
                        continue
                    toks = line.strip().split()
                    res[key]['irrelevant'].append(toks[1:])
    return res

if __name__ == '__main__':
    datapath = sys.argv[1]
    trainsize = sys.argv[2]
    metric = sys.argv[3]
    if metric == 'mu':
        pos = -3
    elif metric == 'theta4':
        pos=-1
    res = get_data(datapath)
    keys, diffs = [], []
    for k in res:
        keys.append(k)
        assert res[k]['adversarial'][pos][0] == metric
        diffs.append((1 - float(res[k]['adversarial'][pos][1])) +  float(res[k]['ethical'][pos][1])) #theta: (1-adv) + (ethical - 0)
    best = np.argsort(diffs)[:5]
    best_keys = [keys[idx] for idx in best]
    best_lr = [bk.split('_')[0] for bk in best_keys]
    best_epoch = [bk.split('_')[1] for bk in best_keys]
    assert len(best_lr) == len(best_epoch) == 5
    best_models = []
    for idx in range(len(best_lr)):
        best_models.append(f'roberta_US_{best_lr[idx]}_{trainsize}T{best_epoch[idx]}E')
    print(' '.join(best_models)) 