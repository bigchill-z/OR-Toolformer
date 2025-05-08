from unsloth import FastLanguageModel
import torch
from tqdm import tqdm
import pandas as pd

max_seq_length = 5120 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
model_path = "YOUR_MODEL"
data_path = "YOUR_DATA"
output_path = "YOUR_OUTPUT"
lora_path = "YOUR_LORA"

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = model_path,
#     max_seq_length = max_seq_length,
#     dtype = dtype,
#     load_in_4bit = load_in_4bit,
#     # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
# )




model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=lora_path,  # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)
# 5tools
tool_json = [{'name': 'Solve IP', 'description': 'useful when you need to solve an Integer Programming (IP: all decision variables are required to take integer values, and both the objective function and constraints are linear.) problem defined by an objective function, constraints, variable bounds, and binary variables.', 'parameters': {'type': 'object', 'properties': {'goal': {'type': 'string', 'description': "the goal of the IP, which can be either 'Maximize' or 'Minimize'.", 'enum': ['Maximize', 'Minimize']}, 'objective_function': {'type': 'string', 'description': "the objective function expressed in lp file format (e.g., 'obj: 8 x1 + 2 x2 + 3 x3')."}, 'constraints': {'type': 'array', 'description': "a list of constraints expressed in lp file format (e.g., ['c1: 4 x1 + 3 x2 <= 28', 'c2: 3 x1 + 2 x2 <= 19', 'c3: x2 + 3 x3 <= 14']).", 'items': {'type': 'string'}}, 'variable_bounds': {'type': 'array', 'description': "a list of variable bounds expressed in lp file format (e.g., ['x1 >= 0', 'x2 >= 0']).", 'items': {'type': 'string'}}, 'variable_binaries': {'type': 'array', 'description': "a list of binary variables (e.g., ['x3']).", 'items': {'type': 'string'}}}, 'required': ['goal', 'objective_function', 'constraints', 'variable_bounds', 'variable_binaries']}}, {'name': 'Solve LP', 'description': 'useful when you need to solve a Linear Programming (LP: find the best outcome in a model with linear relationships, subject to a set of constraints.) problem defined by an objective function, constraints, and variable bounds.', 'parameters': {'type': 'object', 'properties': {'goal': {'type': 'string', 'description': "the goal of the LP, which can be either 'Maximize' or 'Minimize'.", 'enum': ['Maximize', 'Minimize']}, 'objective_function': {'type': 'string', 'description': "the objective function expressed in lp file format (e.g., 'obj: 8 x + 15 y')."}, 'constraints': {'type': 'array', 'description': "a list of constraints expressed in lp file format (e.g., ['c1: 10 x + 15 y <= 3000', 'c2: x + y <= 250']).", 'items': {'type': 'string'}}, 'variable_bounds': {'type': 'array', 'description': "a list of variable bounds expressed in lp file format (e.g., ['x >= 0', 'y >= 0']).", 'items': {'type': 'string'}}}, 'required': ['goal', 'objective_function', 'constraints', 'variable_bounds']}}, {'name': 'Solve MILP', 'description': 'useful when you need to solve a Mixed Integer Linear Programming (MILP: the objective function and constraints are linear, but some of the decision variables are restricted to integer values, while others can be continuous.) problem defined by an objective function, constraints, variable bounds, binary variables, and integer variables.', 'parameters': {'type': 'object', 'properties': {'goal': {'type': 'string', 'description': "the goal of the MILP, which can be either 'Maximize' or 'Minimize'.", 'enum': ['Maximize', 'Minimize']}, 'objective_function': {'type': 'string', 'description': "the objective function expressed in lp file format (e.g., 'obj: 3 x1 + 5 x2 + x3 + x4')."}, 'constraints': {'type': 'array', 'description': "a list of constraints expressed in lp file format (e.g., ['c1: 2 x1 + x2 + x4 <= 18.5', 'c2: x1 + 2 x2 <= 15', 'c3: x2 + x3 <= 8.5']).", 'items': {'type': 'string'}}, 'variable_bounds': {'type': 'array', 'description': "a list of variable bounds expressed in lp file format (e.g., ['x1 >= 0', 'x2 >= 0', 'x3 >= 0']).", 'items': {'type': 'string'}}, 'variable_binaries': {'type': 'array', 'description': "a list of binary variables (e.g., ['x4']).", 'items': {'type': 'string'}}, 'variable_integers': {'type': 'array', 'description': "a list of integer variables (e.g., ['x1', 'x2']).", 'items': {'type': 'string'}}}, 'required': ['goal', 'objective_function', 'constraints', 'variable_bounds', 'variable_binaries', 'variable_integers']}}, {'name': 'Solve TSP with Distance Matrix', 'description': 'useful when you need to solve a Traveling Salesman Problem (TSP: find the shortest possible route that visits each location exactly once and returns to the origin location.) using a distance matrix representation of the nodes.', 'parameters': {'type': 'object', 'properties': {'num_nodes': {'type': 'integer', 'description': 'the number of nodes to visit.'}, 'matrix_type': {'type': 'string', 'description': 'the type of the distance matrix, which can be one of the following: "LOWER_DIAG_ROW", "FULL_MATRIX", "UPPER_DIAG_ROW".', 'enum': ['LOWER_DIAG_ROW', 'FULL_MATRIX', 'UPPER_DIAG_ROW']}, 'matrix_data': {'type': 'array', 'description': 'a list representing the distance values according to the specified matrix type.', 'items': {'type': 'array', 'items': {'type': 'number'}}}}, 'required': ['num_nodes', 'matrix_type', 'matrix_data']}}, {'name': 'Solve MF with Matrix', 'description': 'useful when you need to solve a Maximum Flow (MF: find the greatest possible flow of resources from a source node to a sink node in a network, while respecting the capacities of the edges.) problem using a matrix representation of the capacities.', 'parameters': {'type': 'object', 'properties': {'matrix': {'type': 'array', 'description': 'a 2D array where non-zero values represent the capacity of the edges between nodes. The element at position [i][j] indicates the capacity from node i to node j.', 'items': {'type': 'array', 'items': {'type': 'integer'}}}}, 'required': ['matrix']}}] 
# 3tools
tool_json = [{'name': 'Solve IP', 'description': 'useful when you need to solve an Integer Programming (IP: all decision variables are required to take integer values, and both the objective function and constraints are linear.) problem defined by an objective function, constraints, variable bounds, and binary variables.', 'parameters': {'type': 'object', 'properties': {'goal': {'type': 'string', 'description': "the goal of the IP, which can be either 'Maximize' or 'Minimize'.", 'enum': ['Maximize', 'Minimize']}, 'objective_function': {'type': 'string', 'description': "the objective function expressed in lp file format (e.g., 'obj: 8 x1 + 2 x2 + 3 x3')."}, 'constraints': {'type': 'array', 'description': "a list of constraints expressed in lp file format (e.g., ['c1: 4 x1 + 3 x2 <= 28', 'c2: 3 x1 + 2 x2 <= 19', 'c3: x2 + 3 x3 <= 14']).", 'items': {'type': 'string'}}, 'variable_bounds': {'type': 'array', 'description': "a list of variable bounds expressed in lp file format (e.g., ['x1 >= 0', 'x2 >= 0']).", 'items': {'type': 'string'}}, 'variable_binaries': {'type': 'array', 'description': "a list of binary variables (e.g., ['x3']).", 'items': {'type': 'string'}}}, 'required': ['goal', 'objective_function', 'constraints', 'variable_bounds', 'variable_binaries']}}, {'name': 'Solve LP', 'description': 'useful when you need to solve a Linear Programming (LP: find the best outcome in a model with linear relationships, subject to a set of constraints.) problem defined by an objective function, constraints, and variable bounds.', 'parameters': {'type': 'object', 'properties': {'goal': {'type': 'string', 'description': "the goal of the LP, which can be either 'Maximize' or 'Minimize'.", 'enum': ['Maximize', 'Minimize']}, 'objective_function': {'type': 'string', 'description': "the objective function expressed in lp file format (e.g., 'obj: 8 x + 15 y')."}, 'constraints': {'type': 'array', 'description': "a list of constraints expressed in lp file format (e.g., ['c1: 10 x + 15 y <= 3000', 'c2: x + y <= 250']).", 'items': {'type': 'string'}}, 'variable_bounds': {'type': 'array', 'description': "a list of variable bounds expressed in lp file format (e.g., ['x >= 0', 'y >= 0']).", 'items': {'type': 'string'}}}, 'required': ['goal', 'objective_function', 'constraints', 'variable_bounds']}}, {'name': 'Solve MILP', 'description': 'useful when you need to solve a Mixed Integer Linear Programming (MILP: the objective function and constraints are linear, but some of the decision variables are restricted to integer values, while others can be continuous.) problem defined by an objective function, constraints, variable bounds, binary variables, and integer variables.', 'parameters': {'type': 'object', 'properties': {'goal': {'type': 'string', 'description': "the goal of the MILP, which can be either 'Maximize' or 'Minimize'.", 'enum': ['Maximize', 'Minimize']}, 'objective_function': {'type': 'string', 'description': "the objective function expressed in lp file format (e.g., 'obj: 3 x1 + 5 x2 + x3 + x4')."}, 'constraints': {'type': 'array', 'description': "a list of constraints expressed in lp file format (e.g., ['c1: 2 x1 + x2 + x4 <= 18.5', 'c2: x1 + 2 x2 <= 15', 'c3: x2 + x3 <= 8.5']).", 'items': {'type': 'string'}}, 'variable_bounds': {'type': 'array', 'description': "a list of variable bounds expressed in lp file format (e.g., ['x1 >= 0', 'x2 >= 0', 'x3 >= 0']).", 'items': {'type': 'string'}}, 'variable_binaries': {'type': 'array', 'description': "a list of binary variables (e.g., ['x4']).", 'items': {'type': 'string'}}, 'variable_integers': {'type': 'array', 'description': "a list of integer variables (e.g., ['x1', 'x2']).", 'items': {'type': 'string'}}}, 'required': ['goal', 'objective_function', 'constraints', 'variable_bounds', 'variable_binaries', 'variable_integers']}}] 


system = f"""You are a mathematical modeling expert and a professor of operations research at a top university. You are very good at solving various operations research problems. When solving operations research problems, you will first conduct mathematical modeling (if necessary), and then use appropriate tools to obtain the optimal solution. Users can provide you with problem descriptions in plain text, but any problem must be solved using the correct tools. 

<tools>{tool_json}</tools>

When you need to use a tool, please follow this format: <use_tool>tool_name(param1, param2, ...)</use_tool>

Always use the data provided in the problem description to avoid fabricating any incorrect data. Remember to pass in accurate and complete parameters to the tool without omission!"""
print(system)
def get_answer(text):
    try:
        text = tokenizer.apply_chat_template([
                        {'role': 'system', 'content':system},
                        {'role': 'user', 'content':text.strip('\n').strip()+'\n\nYou need to use a suitable tool to solve this problem, rather than solving it directly. Let’s think step by step.'}],
                        tokenize=False, 
                        add_generation_prompt=True).replace('''Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n''','')
        inputs = tokenizer([text], return_tensors = "pt").to("cuda")
        
        outputs = model.generate(**inputs, max_new_tokens = max_seq_length, use_cache = True,do_sample=False,top_p=0.95)
        return tokenizer.batch_decode(outputs)[0].split('assistant')[-1]
    except:
        return 'error'


df = pd.read_excel(data_path)
df = df[df['name']!='NLP4LP'].reset_index(0)
# df.drop_duplicates(subset='user', keep='first', inplace=True, ignore_index=True)
# 初始化 tqdm 以与 pandas 配合使用

# 初始化 tqdm 以与 pandas 配合使用
# tqdm.pandas(desc="Progress")  # 设置进度条描述信息
result = {'question':[],'unsloth-SFT':[]}
for i in tqdm(range(len(df))):
    q = df['user'][i]
    result['question'].append(q)
    result['unsloth-SFT'].append(get_answer(q))
    pd.DataFrame(result).to_excel(f'{output_path}/unsloth_sft_result.xlsx',index=False)
pd.DataFrame(result).to_excel(f'{output_path}/unsloth_sft_result.xlsx',index=False)