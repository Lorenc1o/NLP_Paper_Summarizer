#!/bin/bash

# Run all experiments

# Example:
# ./run_experiments.sh

# Train and test linear model
./experiment.sh --train True --test True --type linear --train_data preprocessing/data/dialogueSum/stories/train.pt --val_data preprocessing/data/dialogueSum/stories/validation.pt --test_data preprocessing/data/dialogueSum/stories/test.pt --train_output output/ --model_name model_linear.pt --load output/model_linear.pt --epochs 100 --batch_size 32 --n_sents 3 --verbose False --output_file output/linear.json

# Train and test transformer model
./experiment.sh --train True --test True --type transformer --train_data preprocessing/data/dialogueSum/stories/train.pt --val_data preprocessing/data/dialogueSum/stories/validation.pt --test_data preprocessing/data/dialogueSum/stories/test.pt --train_output output/ --model_name model_transformer.pt --load output/model_transformer.pt --epochs 100 --batch_size 32 --n_sents 3 --verbose False --output_file output/transformer.json