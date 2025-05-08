import argparse
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
def parse_args():
    parser = argparse.ArgumentParser(description="Script to set parameters for model training")

    # Adding arguments with default values
    parser.add_argument('--max_seq_length', type=int, default=3072,
                        help="Maximum sequence length (default: 3072)")
    parser.add_argument('--dtype', type=str, default=None,
                        help="Data type (default: None for auto detection, use 'float16' for Tesla T4, 'bfloat16' for Ampere+)")
    parser.add_argument('--load_in_4bit', type=bool, default=True,
                        help="Use 4bit quantization to reduce memory usage (default: True)")
    parser.add_argument('--model_path', type=str, default="/gemini/pretrain",
                        help="Path to the model (default: /gemini/pretrain)")
    parser.add_argument('--data_path', type=str, default="/gemini/data-3/Merge_Chat5_for_Train.xlsx",
                        help="Path to the training data (default: /gemini/data-1/Merge_Chat5_for_Train.xlsx)")
    parser.add_argument('--output_path', type=str, default="/gemini/output",
                        help="Path to output directory (default: /gemini/output)")

    # Parse the arguments
    args = parser.parse_args()

    return args
args = parse_args()

# Access the arguments
max_seq_length=args.max_seq_length
dtype=args.dtype
load_in_4bit=args.load_in_4bit
model_path=args.model_path
data_path=args.data_path
output_path=args.output_path

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

def formatting_prompts_func(examples):
    systems = examples["system"]
    users = examples["user"]
    assistants = examples["assistant"]
    texts = []
    # for instruction, input, output in zip(instructions, inputs, outputs):
    #     # Must add EOS_TOKEN, otherwise your generation will go on forever!
    #     text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
    #     texts.append(text)
    # return { "text" : texts, }
    for system, user, assistant in zip(systems, users, assistants):

        text = tokenizer.apply_chat_template([
                {'role': 'system', 'content':system},
                {'role': 'user', 'content':user},
                {'role': 'assistant', 'content':'\n'.join([i.replace('**','').strip('*- ').strip() for i in assistant.split('\n')])}],
                tokenize=False, 
                add_generation_prompt=False)
        texts.append(text.replace('''Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n''',''))
        
    return { "text" : texts, }

from datasets import load_dataset
import pandas as pd
df= pd.read_excel(data_path)
df.to_csv(f'{output_path}/GRPO_Chat5_1000.csv',index=False)
dataset = load_dataset("csv", data_files=f"{output_path}/GRPO_Chat5_1000.csv")

dataset = dataset.map(formatting_prompts_func, batched = True,)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset['train'],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 8,
        
        num_train_epochs = 2,# warmup_ratio for full training runs!
        warmup_steps = 25,
        # max_steps = 60,

        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = output_path,
        report_to = "none", # Use this for WandB etc
    ),
)
trainer_stats = trainer.train()

model.save_pretrained(f"{output_path}/lora_model") # Local saving
tokenizer.save_pretrained(f"{output_path}/lora_model")