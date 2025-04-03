set -x

GPUS=${GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-2}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34239
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='/fs-computility/ai-shen/xueyingyi/grpo_test'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 2
# batch size per gpu: 4
# gradient accumulation steps: 2
# total batch size: 16
# epoch: 1
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune_cot_grpo.py \
  --model_name_or_path "/fs-computility/ai-shen/xueyingyi/model/cot_box_text" \
  --ref_model_path "/fs-computility/ai-shen/xueyingyi/model/cot_box_text" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --use_grpo True \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "/fs-computility/ai-shen/xueyingyi/cot_picture/data/train_data.jsonl" \
  --meta_path_eval "/fs-computility/ai-shen/xueyingyi/cot_picture/data/eval_data.jsonl" \
  --overwrite_output_dir True \
  --load_best_model_at_end True \
  --metric_for_best_model "avg_text_similarity" \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone False \
  --use_llm_lora 16 \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 10 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "steps" \
  --save_strategy "steps" \
  --save_steps 200 \
  --eval_steps 100 \
  --save_total_limit 2 \
  --learning_rate 1e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
