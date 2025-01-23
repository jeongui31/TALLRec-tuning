CUDA_ID=$1
output_dir=$2
dataset=$3
base_model="./model"
test_data="./data/$3/test.json"

echo $output_dir
CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
    --base_model $base_model \
    --lora_weights $output_dir \
    --test_data_path $test_data \
    --result_json_data $2.json