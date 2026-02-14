# SAMChat: Introducing Chain-of-Thought Reasoning and GRPO to a Multimodal Small Language Model for Small-Scale Remote Sensing

## IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing

In this work, a lightweight multimodal language model termed SAMChat is introduced, specifically adapted to analyze remote sensing imagery in secluded areas, including challenging missile launch sites. A new dataset, SAMData, was compiled by verifying hundreds of aerial images through expert review, and subtle military installations were highlighted via detailed captions. Supervised fine-tuning on a 2B-parameter open-source MLLM with chain-of-thought reasoning annotations was performed, enabling more accurate and interpretable explanations. In addition, group relative policy optimization was leveraged to enhance the model’s ability to detect critical domain-specific cues—such as defensive layouts and key military structures—while minimizing false positives on civilian scenes. Through empirical evaluations, it has been shown that SAMChat significantly outperforms both larger, general-purpose multimodal models and existing remote sensing-adapted approaches on open-ended captioning and classification metrics. Over 80% recall and 98% precision were achieved on the newly proposed SAMData benchmark, underscoring the potency of targeted fine-tuning and reinforcement learning in specialized real-world applications. 

### [Paper (arXiv)](https://arxiv.org/abs/2505.07984) | [Paper (IEEExplore)](https://ieeexplore.ieee.org/document/11267210)

### Dataset

<table>
  <thead>
    <tr style="text-align: right;">
      <th>dataset</th>
      <th>purpose</th>
      <th>link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>SAMData-300-Train</td>
      <td>Training</td>
      <td><a href="https://huggingface.co/datasets/aybora/SAMData-300-Train">aybora/SAMData-300-Train</a></td>
    </tr>
    <tr>
      <td>SAMData-300-Test</td>
      <td>Testing</td>
      <td><a href="https://huggingface.co/datasets/aybora/SAMData-300-Test">aybora/SAMData-300-Test</a></td>
    </tr>
  </tbody>
</table>

### Models

<table>
  <thead>
    <tr style="text-align: right;">
      <th>model</th>
      <th>type</th>
      <th>link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>SAMChat-Base</td>
      <td>instant</td>
      <td><a href="https://huggingface.co/aybora/Qwen2-VL-SAMChat-Base">aybora/Qwen2-VL-SAMChat-Base</a></td>
    </tr>
    <tr>
      <td>SAMChat-Distill</td>
      <td>reasoning (sft only)</td>
      <td><a href="https://huggingface.co/aybora/Qwen2-VL-SAMChat-Distill">aybora/Qwen2-VL-SAMChat-Distill</a></td>
    </tr>
    <tr>
      <td>SAMChat-Zero</td>
      <td>reasoning (rl only)</td>
      <td><a href="https://huggingface.co/aybora/Qwen2-VL-SAMChat-Zero">aybora/Qwen2-VL-SAMChat-Zero</a></td>
    </tr>
    <tr>
      <td>SAMChat-R1</td>
      <td>reasoning (sft+rl)</td>
      <td><a href="https://huggingface.co/aybora/Qwen2-VL-SAMChat-R1">aybora/Qwen2-VL-SAMChat-R1</a></td>
    </tr>
  </tbody>
</table>

### Installation

For best reproducibility, we suggest you to generate three different environments, one each for finetuning, GRPO training and evaluation.

For SFT:

```shell
git clone https://github.com/aybora/SAMChat
conda env create -f environment.yaml
conda activate sft
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```

For GRPO:

```shell
git clone https://github.com/aybora/SAMChat
conda create -n grpo python=3.10 -y
conda activate grpo
cd ~/SAMChat/grpo
pip3 install -e ".[dev]"
pip3 install wandb==0.18.3
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
```

For Evaluation:

```shell
conda create -n eval python==3.10 -y
conda activate eval
cd ~/SAMChat/eval 
pip install -r requirements.txt
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
```

### Training (SFT)

First download training set folder from our [Huggingface Repo](https://huggingface.co/datasets/aybora/SAMData-300-Train).

For reproducing Base and Distill models, you may follow the sample script below, which works on one node with 4 x H100s or A100s. Use sam_300_inst.json for the Base model and sam_300_reasoning_inst.json for the Distill model.

```shell

conda activate sft
cd ./SAMChat/sft/

MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=16
NUM_DEVICES=4
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export PYTHONPATH=src:$PYTHONPATH

deepspeed --master_port 29400 src/training/train.py \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path "Your path to sam_300_inst.json or sam_300_reasoning_inst.json file" \
    --image_folder "Your path to SAMData-300-Train folder" \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --tune_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/qwen_sam_300 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((512 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 5 \
    --dataloader_num_workers 4
```
### Training (GRPO)

Below script works on at least one node with 4 x H100s or A100s (65-80 GB). Use "Qwen/Qwen2-VL-2B-Instruct" for reproducing the Zero model and "aybora/Qwen2-VL-SAMChat-Distill" for the R1 model.

```shell
export WANDB_RUN_NAME=Qwen-VL-2B-GRPO-$(date +%Y-%m-%d-%H-%M-%S)

torchrun \
    --nproc_per_node="$GPUS_PER_NODE" \
    --nnodes="$SLURM_NNODES" \
    --node_rank="$SLURM_NODEID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    --rdzv_id $SLURM_JOB_ID \
    src/open_r1/grpo.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir checkpoints/${WANDB_RUN_NAME} \
    --model_name_or_path aybora/Qwen2-VL-SAMChat-Distill \
    --dataset_name aybora/VHM_dataset_grpo \
    --max_prompt_length 8192 \
    --max_completion_length 8192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 true \
    --beta 0.001 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 2359296 \
    --save_total_limit 6 \
    --num_train_epochs 64 \
    --num_generations 4 \
    --save_steps 100 \
    --run_name $WANDB_RUN_NAME
```
You may need to adjust some of the parameters (MASTER_ADDR, GPUS_PER_NODE etc.) depending on your multi-gpu, multi-node setting. 

### Evaluation

First download datasets eval folder from our [Huggingface Repo](https://huggingface.co/datasets/aybora/SAMData-300-Test).

To evaluate our, or your reproduced model, you may use the script below:

```shell

DATA_ROOT="Your path to SAMData-300-Test folder"
OUTPUT_DIR="Your path to eval log files"
MODEL_PATH=aybora/Qwen2-VL-SAMChat-R1 #or your own local model

python samchat_infer_eval.py \
  --model $MODEL_PATH \
  --test_folder $DATA_ROOT \
  --output_dir $OUTPUT_DIR \
  --num_gpus 1 #supports multi-gpu

```

### Acknowledgements

Our work is derived from [Qwen2-VL](https://github.com/QwenLM/Qwen2.5-VL) for the base model, [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune) for the forked main sft code, [open-r1-multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal) for the forked main grpo code. We appreciate all of these great works.

### Citation

If you find this code useful for your research, please consider citing our works:

```bibtex
@article{koksal2025samchat,
  title={SAMChat: Introducing Chain-of-Thought Reasoning and GRPO to a Multimodal Small Language Model for Small-Scale Remote Sensing},
  author={K{\"o}ksal, Aybora and Alatan, A Ayd{\i}n},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={19},
  pages={795--804},
  year={2025},
  publisher={IEEE}
}
```

```bibtex
@article{koksal2025tinyrs,
  title={Tinyrs-r1: Compact vision language model for remote sensing},
  author={K{\"o}ksal, Aybora and Alatan, A Ayd{\i}n},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2025},
  publisher={IEEE}
}
```

If you are interested in this work, you may find the following work also useful:

```bibtex
@inproceedings{koksal2025few,
  title={Few-Shot Vision-Language Reasoning for Satellite Imagery via Verifiable Rewards},
  author={K{\"o}ksal, Aybora and Alatan, A Ayd{\i}n},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6901--6910},
  year={2025}
}
```
