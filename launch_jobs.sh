#!/bin/bash

# Define the list of integers
seeds=(42)
#seeds=(42 65 82 105 122 132 142 152 162 172) 
#seeds=(142 152 162 172)
#seeds=(182 192 202 212 222) 
# seeds=(132 202 212 222) 

# Loop through the integers
for seed in "${seeds[@]}"
do
    # Submit the job submit script with the current integer as input
    qsub -v SEED=$seed submit.sh 
    # qsub -v FOO="hello",BAR="world" example.sh
done
