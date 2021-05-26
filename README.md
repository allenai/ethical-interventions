# [Ethical-Advice Taker:Do Language Models Understand Natural Language Interventions?]() (ACL 2021 Findings)

[Jieyu Zhao](https://jyzhao.net/), [Daniel Khashabi](https://danielkhashabi.com/), [Tushar Khot](https://allenai.org/team/tushark), [Ashish Sabharwal](https://allenai.org/team/ashishs), and [Kai-Wei Chang](http://web.cs.ucla.edu/~kwchang/)

## Linguistic Ethical Interventions (LEI) 🌺 
### Data
In the paper, we propose the LEI challenge based on the [UnQover] dataset by adding interventions to the QA example to verify if existing models can understand and follow the instructions we provided. We provide the training and eval dataset used in our paper:
- **Easiest way**: You can download the data we used for train/test in the paper [here](/data).
- **From Scratch** 
    - We add interventions to a subset of UnQover (covers religion, ethnicity and gender). Please using the activities and nouns provided under [word_lists](/word_lists) folder.
    - The interventions are under [interventions](/interventions) folder. 
    - Run the script `python xxx.py`.

### Model & Eval: 
Our model is based on RoBERTa-SQuAD model, which can be downloaded from [UnQover]() repo ([model link]()).
> remember to install the required packages as in requirement.txt

To fine tune the above model on our LEI dataset, simply run `./runbeaker.sh 1e-5 data outputs 3 12 3072 "--doadversarial --doirrelevant" U_ mu 5`.