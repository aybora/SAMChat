# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration

#from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


def accuracy_reward(completions, solution, **kwargs):
    """Custom reward function based on keyword presence and a solution flag."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    keywords = ["military", "missile", "silo"]
    
    for content, sol in zip(contents, solution):
        # Extract the text within <answer> tags; if not found, use the full content.
        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        student_answer = answer_match.group(1).strip() if answer_match else content.strip()
        
        # Check if any keyword is present (case-insensitive).
        found_keyword = any(keyword in student_answer.lower() for keyword in keywords)
        
        # Decide the reward based on the solution flag.
        if str(sol).strip() == "1":
            # If solution is 1, reward is 1 if any keyword is found.
            reward = 1.0 if found_keyword else 0.0
        elif str(sol).strip() == "0":
            # If solution is 0, reward is 0 if any keyword is found.
            reward = 0.0 if found_keyword else 1.0
        else:
            # If the solution flag is neither 1 nor 0, default to 0 reward.
            reward = 0.0
        
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution flag: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    dataset = load_dataset('imagefolder', data_dir=script_args.dataset_name)

    QUESTION_TEMPLATE = "{Question} Output the thinking process in <reasoning> </reasoning> and final answer (summary of thinking) in <answer> </answer> tags."

    def make_conversation_image(example):
            return {
                "prompt": [
                    {
                        "role": "system", 
                        "content": [{"type": "text", "text": "You are an expert to analyze the image and provide useful information for users."}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["question"])},
                        ],
                    },
                ],
            }

    dataset = dataset.map(make_conversation_image)

    trainer_cls = Qwen2VLGRPOTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
