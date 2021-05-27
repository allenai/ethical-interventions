# [Ethical-Advice Taker:Do Language Models Understand Natural Language Interventions?]() (ACL 2021 Findings)

[Jieyu Zhao](https://jyzhao.net/), [Daniel Khashabi](https://danielkhashabi.com/), [Tushar Khot](https://allenai.org/team/tushark), [Ashish Sabharwal](https://allenai.org/team/ashishs), and [Kai-Wei Chang](http://web.cs.ucla.edu/~kwchang/)

## Linguistic Ethical Interventions (LEI) 🌺 
### Data
In the paper, we propose the LEI challenge based on [UnQover](https://github.com/allenai/unqover) by adding interventions to the QA example to verify if existing models can understand and follow the instructions we provided. Here we briefly talk about the training and test dataset used in our paper:
- **Easiest way**: You can download the data we used for train/test in the paper [here](/data). It contains the data for both the specified and underspecified cases.  
- **From Scratch** 
    - We add interventions to a subset of UnQover (covers religion, ethnicity and gender). Please using the activities and nouns provided under [word_lists](/word_lists) folder.
    - The interventions are under [interventions](/interventions) folder. 
    - For underspecified case:
        - generate the questions: 
        1. follow the instructions in [UnQover](https://github.com/allenai/unqover).
        ``` 
        TYPE=slot_act_map
        SUBJ=mixed_gender
        SLOT=gender_noact
        ACT=occupation_rev1
        FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}
        python3 -m templates.generate_underspecified_templates --template_type ${TYPE} \
        --subj $SUBJ --act $ACT --slot $SLOT \
        --output ./data/${FILE}.source.json
        ```
        2. Do the predictions. The interventions will be added to the QA context.
        ```
        intv_type=ethical
	    python3 -u -m qa_hf.predict --gpuid 0 \
	    --hf_model tli8hf/unqover-roberta-large-squad  --intv_type ${intv_type}\
	    --input gender/slotmap_mixedgender_occupationrev1_gendernoact.source.json --output ./data/gender/robertalarge_gender-${intv_type}.output.json --attribute gender --converted 0
        ```
    - For specified case (only for gender):
        1. Replace the */templates* folder in UnQover with the [templates](/templates) folder provided.
        2. Repeat the above procedure 


### Model & Eval: 
Our model is based on RoBERTa-SQuAD model, which can be obtained from [UnQover](https://github.com/allenai/unqover) repo (See Section 2).
> remember to install the required packages as in requirement.txt

To fine tune the above model on our LEI dataset, simply run `./run.sh 1e-5 data outputs 64 4 4 1 "--doadversarial --doirrelevant --squad" U_ US theta4 1 0.5 1`. Note that this also includes the model selection part, and feel free to comment that out.