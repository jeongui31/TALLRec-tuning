echo $1, $2, $3
seed=$2
dataset=$3
output_dir="./output/$3/$3"
base_model="./model"
train_data="./data/$3/train.json"
val_data="./data/$3/valid.json"
instruction_model="./alpaca-lora-7B"

for lr in 1e-4
do
    for dropout in 0.05
    do
        for sample in 16
        do
            #mkdir -p $output_dir
            echo "lr: $lr, dropout: $dropout , seed: $seed, sample: $sample"
            CUDA_VISIBLE_DEVICES=$1 python -u finetune_rec_origin.py \
                --base_model $base_model \
                --train_data_path $train_data \
                --val_data_path $val_data \
                --output_dir "${output_dir}_${seed}_${sample}" \
                --batch_size 128 \
                --micro_batch_size 32 \
                --num_epochs 200 \
                --learning_rate $lr \
                --cutoff_len 364 \
                --lora_r 8 \
                --lora_alpha 16 \
                --lora_dropout $dropout \
                --lora_target_modules '[q_proj,v_proj]' \
                --train_on_inputs \
                --group_by_length \
                --resume_from_checkpoint $instruction_model \
                --sample $sample \
                --seed $2
        done
    done
done
