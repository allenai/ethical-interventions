#!/bin/bash

#re-evaluate the model based on the beaker predictions 

GENDEREXTRADATA="$1"
OUTPUTS="$2"
NUM_TRAIN="$3"
BATCH_SIZE="$4"
NUM_GPUS="$5"
NUM_CHECKPOINTS=${6:-5}
WITH=${7:-'--doadversarial --doirrelevant --squad'}
FLAG=${8:-' '}
METRICS=${9:-mu}
BEAKERSAVED=${10}
TOP5=${11}
for f in ${TOP5[@]}
do 
  echo ${f}
done

genders=("Female" "Male")
for pred in ${BEAKERSAVED}/*.output.json
do
  echo ${pred}
  python3 get_acc.py ${pred}  
done