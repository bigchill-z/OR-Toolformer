# OR-Toolformer: Solving Operations Research Problems with Tool-Augmented Large Language Models
## Requirements

Install the necessary dependencies provided in the requirements.txt.

```bash
pip install -r requirements.txt
```

## Directory Structure

```
├── Dataset/                # Contains training datasets and CombOpt datasets 
├── OpToolGen/              # Scripts for synthesizing data for 7 different problem types 
├── Pred/                   # Code for merging LoRA weights and performing model predictions as well as evaluating model performance 
├── Tools/                  # A collection of 11 different tools designed for solving operations research problems 
├── Train/                  # Scripts for training models using unsloth 
├── requirements.txt        # List of dependencies required for the project 
└── utils.py                # Useful utility functions used across the project
```

