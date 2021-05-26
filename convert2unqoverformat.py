"""
convert the dev_squad_predictions format to Tao's unqover format & remove the interventions; so we can make predictions only for the top 10k dev examples.
"""
import json
import sys

inputfile = sys.argv[1]
outputfile = sys.argv[2]
keepintv = int(sys.argv[3])

data = json.load(open(inputfile, 'r'))

res = {}
cnt = 0
if keepintv == 0:
    print("will remove the interventions")
elif keepintv == 1:
    print("will keep the interventions")
for key, exs in data.items():
    subjs = key.strip().split('|')[2:4]
    intv = key.strip().split('|')[-1].split('_')[0]
    assert exs[0]['context'] == exs[1]["context"]
    pos0 = exs[0]['context'].lower().find(subjs[0])
    pos2 = exs[2]['context'].lower().find(subjs[0])

    if keepintv == 0:
        q0 = {"question":exs[0]['q0']["question"].split('.')[1].strip()}
        q1 = {"question":exs[1]['q1']["question"].split('.')[1].strip()}
        q2 ={"question":exs[2]['q0']["question"].split('.')[1].strip()} 
        q3 ={"question":exs[3]['q1']["question"].split('.')[1].strip()} 
    else:
        q0={"question":exs[0]['q0']["question"]}
        q1={"question": exs[1]['q1']["question"]}
        q2={"question": exs[2]['q0']["question"]}
        q3={"question": exs[3]['q1']["question"]}
    if intv not in res:
        res[intv] = {}
    if pos0 < pos2:

        # print(dict(exs[0]['q0'], **q0))

        res[intv][key] = {'line': exs[0]['line'], 'context':exs[0]['context'], 'q0': dict(exs[0]['q0'], **q0), 'q1': dict(exs[1]['q1'], **q1)}
        keycomponents = key.strip().split('|')
        tmp = keycomponents[:2] + [subjs[1]] + [subjs[0]] + keycomponents[4:]
        key2 = '|'.join(tmp)
        res[intv][key2] ={'line': exs[2]['line'], 'context':exs[2]['context'], 'q0': dict(exs[2]['q0'], **q2), 'q1': dict(exs[3]['q1'], **q3)} 
    else:
        res[intv][key] = {'line': exs[2]['line'], 'context':exs[2]['context'], 'q0': dict(exs[2]['q0'], **q2), 'q1': dict(exs[3]['q1'], **q3)}
        keycomponents = key.strip().split('|')
        tmp = keycomponents[:2] + [subjs[1]] + [subjs[0]] + keycomponents[4:]
        key2 = '|'.join(tmp)
        res[intv][key2] ={'line': exs[0]['line'], 'context':exs[0]['context'], 'q0': dict(exs[0]['q0'], **q0), 'q1': dict(exs[1]['q1'], **q1)} 
    cnt += 4

for intv in res:
    json.dump(res[intv], open(outputfile+"_"+intv+".source.json", 'w'), indent=4)
    
