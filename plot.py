"""
plot bias for all the checkpoints
"""

from matplotlib import pyplot as plt 
import os,sys,json
import numpy as np
import glob

#[ethical, adversarial, irrelevant]
mu_taos_mu = {"religion":[0.24714, 0.21126, 0.1801], "ethnicity":[0.14523, 0.1748, 0.15563], "gender": [0.469369, 0.43359, 0.34181]}

def get_data(datapth, cat):
    print(f"reading *.log.txt data from {datapath}")
    checkpoints = []
    res= {}
    for infile in glob.glob(f"{datapath}/*.log.txt"):
        ckp = infile.strip().split('-')[-1].split('.')[0]
        if not ckp.isnumeric():
            continue
        checkpoints.append(int(ckp))
        dashcount = 0
        res[int(ckp)] = {'adversarial': [], 'ethical': [], 'irrelevant':[]}
        with open(infile, 'r') as f:
            for line in f.readlines():
                if line.startswith('----'): #starting point
                    dashcount += 1
                    continue
                if dashcount  == 1:
                    toks = line.strip().split('\t')
                    toks = list(map(lambda x: x.strip(), toks))
                    res[int(ckp)]['ethical'].append(toks)
                if dashcount == 3:
                    toks = line.strip().split('\t')
                    toks = list(map(lambda x: x.strip(), toks))
                    res[int(ckp)]['adversarial'].append(toks)
                if dashcount == 5:
                    toks = line.strip().split('\t')
                    toks = list(map(lambda x: x.strip(), toks))
                    res[int(ckp)]['irrelevant'].append(toks)
                if dashcount == 7:
                    if line.startswith("intervention"):
                        continue
                    toks = line.strip().split()
                    res[int(ckp)]['ethical'].append(toks[1:])
                if dashcount == 8:
                    if line.startswith("intervention"):
                        continue
                    toks = line.strip().split()
                    res[int(ckp)]['adversarial'].append(toks[1:])
                if dashcount == 9:
                    if line.startswith("intervention"):
                        continue
                    toks = line.strip().split()
                    res[int(ckp)]['irrelevant'].append(toks[1:])
    return res

def plotfig(results, cat='religion', metric='mu', datapath='output'):
    print(f"plotting for {cat} category")
    plt.close('all')
    linestyle = "solid"
    pos = -2
    if metric == 'eta':
        pos = -1
    
    x = []
    adv_mus = []
    eth_mus = []
    adv_subjs = []
    eth_subjs = []
    irr_mus, irr_subjs = [], []
    
    for i in sorted(results):
        advs = res[i]['adversarial']
        eths = res[i]['ethical']
        irrs = res[i]['irrelevant']
        x.append(int(i))
        adv_mus.append(float(advs[pos][1]))
        eth_mus.append(float(eths[pos][1]))
        irr_mus.append(float(irrs[pos][1]))
        adv_subjs.append({advs[m][0]: float(advs[m][1]) for m in range(len(advs) - 2)})
        eth_subjs.append({eths[m][0]: float(eths[m][1]) for m in range(len(eths) - 2)})
        irr_subjs.append({irrs[m][0]: float(irrs[m][1]) for m in range(len(irrs) - 2)})
    

    plt.plot(x, eth_mus, label='ethical ours', linestyle = linestyle)
    plt.plot(x, adv_mus, label='adversarial ours', linestyle = linestyle)
    plt.plot(x, irr_mus, label='irrelevant ours', linestyle = linestyle)
    plt.plot(x, [mu_taos_mu[cat][0]]*len(x), label='ethical baseline', linestyle="dotted")
    plt.plot(x, [mu_taos_mu[cat][1]]*len(x), label='adversarial baseline', linestyle="dotted")
    plt.plot(x, [mu_taos_mu[cat][2]]*len(x), label='irrelevant baseline', linestyle="dotted")
    plt.xticks(x, rotation='vertical')
    plt.xlabel("save_checkpoint")
    plt.ylabel(f"{metric} score")
    # plt.legend(bbox_to_anchor=(1.02, 1.05))
    plt.legend()
    plt.savefig(f'{datapath}/{cat}_{metric}.jpg')

if __name__ == '__main__':
    datapath = sys.argv[1]
    cat = sys.argv[2]
    metric = sys.argv[3]
    res = get_data(datapath, cat)
    plotfig(res, cat, metric, datapath)
