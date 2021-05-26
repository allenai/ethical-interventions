import pandas as pd 
import json
import sys

intv_cat = sys.argv[1]
religion_tsv = pd.read_csv(f"./unqover_{intv_cat}_interventions.tsv", sep = '\t')
interventions = ["ethical Interventions", "adversarial Interventions", "irrelevant Interventions"] #, "Irrelevant Interventions-2(change event)"]
res = {}
if intv_cat == 'gender':
    for intervention in interventions:
        res[intervention.split()[0].lower()] = {'none': list(religion_tsv[intervention].values)}

else:
    for intervention in interventions:
        ethical_ints = religion_tsv.groupby(by = ["activity category"])[intervention].apply(list).reset_index(name='ints')
        tmp = {}
        for category, ins in ethical_ints.values:
            tmp[category] = ins
        res[intervention.split()[0].lower()]=tmp

json.dump(res, open(f"{intv_cat}_interventions.json", "w"))