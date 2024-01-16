#!/bin/bash

# Args:
# 1. --train: True if training, False if not
# 2. --test: True if testing, False if not
# 3. --type: linear of transformer
# 4. --train_data: path to training data
# 5. --val_data: path to validation data
# 6. --test_data: path to testing data
# 7. --train_output: path to save training output
# 7. --model_name: path to save model
# 8. --load: path to load model
# 9. --epochs: number of epochs
# 10. --batch_size: batch size
# 11. --n_sents: number of sentences in summary
# 12. --verbose: True if verbose, False if not
# 13. --output_file: path to output file

# Example:
# ./experiment.sh --train True --test True --type linear --train_data preprocessing/data/dialogueSum/stories/train.pt --val_data preprocessing/data/dialogueSum/stories/validation.pt --test_data preprocessing/data/dialogueSum/stories/test.pt --train_output output/model.pt --model_name output/model_linear.pt --load output/model.pt --epochs 10 --batch_size 32 --n_sents 3 --verbose False --output_file output/linear.txt

# Get arguments
OPTS=$(getopt -o '' --long train:,test:,type:,train_data:,val_data:,test_data:,train_output:,model_name:,load:,epochs:,batch_size:,n_sents:,verbose:,output_file: -- "$@")

if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; exit 1 ; fi

eval set -- "$OPTS"

while true; do
    case "$1" in 
        --train ) train="$2"; shift 2 ;;
        --test ) test="$2"; shift 2 ;;
        --type ) type="$2"; shift 2 ;;
        --train_data ) train_data="$2"; shift 2 ;;
        --val_data ) val_data="$2"; shift 2 ;;
        --test_data ) test_data="$2"; shift 2 ;;
        --train_output ) train_output="$2"; shift 2 ;;
        --model_name ) model_name="$2"; shift 2 ;;
        --load ) load="$2"; shift 2 ;;
        --epochs ) epochs="$2"; shift 2 ;;
        --batch_size ) batch_size="$2"; shift 2 ;;
        --n_sents ) n_sents="$2"; shift 2 ;;
        --verbose ) verbose="$2"; shift 2 ;;
        --output_file ) output_file="$2"; shift 2 ;;
        -- ) shift; break ;;
        * ) break ;;
    esac
done

echo "train: $train"
echo "test: $test"
echo "type: $type"
echo "train_data: $train_data"
echo "val_data: $val_data"
echo "test_data: $test_data"
echo "train_output: $train_output"
echo "model_name: $model_name"
echo "load: $load"
echo "epochs: $epochs"
echo "batch_size: $batch_size"
echo "n_sents: $n_sents"
echo "verbose: $verbose"
echo "output_file: $output_file"

# Train
if [ "$train" = "True" ]; then
    echo "Training..."
    python3 train.py --model_type $type --train_loc $train_data --valid_loc $val_data --output_dir $train_output --model_loc $model_name --epochs $epochs --batch_size $batch_size --verbose $verbose
fi

# Test
if [ "$test" = "True" ]; then
    echo "Testing..."
    python3 test.py --model_path $load --test_data $test_data --output_file $output_file --sum_len $n_sents --model_type $type
fi