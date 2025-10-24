export CUDA_VISIBLE_DEVICES=3,5
DATASET=Instruments
DATA_PATH=../data
OUTPUT_DIR=./ckpt/$DATASET/
TIMESTAMP=$(date +"%Y%m%d_%H%M%S") # 获取当前时间戳
RESULTS_FILE=./results/$DATASET/result_${TIMESTAMP}.json # 将时间戳加入文件名
# RESULTS_FILE=./results/$DATASET/xxx.json
CKPT_PATH=./ckpt/$DATASET/

python test.py \
    --gpu_id 0 \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 32 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --index_file .index.json
