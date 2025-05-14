#!/bin/bash

LOG_DIR="drift/data/evaluation/"  # Set your desired log directory
# mkdir -p "$LOG_DIR"  # Ensure the log directory exists

# nohup python3 -m evaluate > evaluation.log 2>&1 & echo $! > save_pid.txt
nohup python3 -m selfcheck --tasks="['csqa', 'cose', 'aqua', 'mathqa']" --results_filename="data/evaluation/baseline/self_check_evaluation_results.json" > evaluation.log 2>&1  &
echo "module1 started with list: ['csqa', 'cose', 'aqua', 'mathqa']"

nohup python3 -m selfcheck --tasks "['medmcqa', 'medqa', 'piqa', 'pubmedqa']" --results_filename "data/evaluation/baseline/self_check_evaluation_results.json" > evaluation.log 2>&1  &
echo "module1 started with list: ['medmcqa', 'medqa', 'piqa', 'pubmedqa']"
