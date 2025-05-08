from gevent import monkey
monkey.patch_all()
import gevent
from gevent.queue import Queue
import sys
import re
import traceback
import random
import time
sys.path.append("./") 
# from process import extract_execute_tool,judge_init_thought
from mypipe.postProcess7.utils import *
from mypipe.Tools.constants import *
from mypipe.Tools.LP import *
# from mypipe.Tools1.graph import *
from mypipe.Tools.MILP import *
from mypipe.Tools.TSP import *
from mypipe.Tools.AP import *
from mypipe.Tools.MF import *
from mypipe.Tools.IP import *
from mypipe.Tools.MCF import *
from mypipe.utils import *
import pandas as pd
from tqdm import tqdm
import pandas as pd
import json
import random
import threading
import concurrent.futures
import re
import copy

from openai import OpenAI

def get_q(data):
    # STYLE #
  # Follow the writing style of excellent operations research teachers when setting questions.
  # # TONE #
  # Accurate, clear, consistent, and easy to understand.
  # # AUDIENCE #
  # Your audience is students who are learning operations research knowledge. Adjust your question so that they can be accurately understood and answered.
    # 5. The expression should be flexible and diverse, avoid repetition, and maintain rich and varied content.
    prompt = f"""# CONTEXT #
You are a professional operations research teacher, skilled in designing problems related to operations research. To help students further master Mixed Integer Linear Programming, we now need you to set a reasonable Mixed Integer Linear Programming problem.
# OBJECTIVE #
<DATA INFORMATION>
INDUSTRY: {random.choice(['Energy', 'Health', 'Retail', 'Environment', 'Education', 'Agriculture', 'Public Utilities', 'Manufacturing', 'Software', 'Construction', 'Entertainment Legal', 'Customer Service', 'Transportation', 'Financial Services'])}
GOAL: {data['forQuestion']['GOAL']}
OBJECTIVE FUNCTION: {data['forQuestion']['OBJECTIVE FUNCTION']}
CONSTRAINTS: {data['forQuestion']['CONSTRAINTS']}
BINARIES: {data['forQuestion']['BINARIES']}
INTEGERS: {data['forQuestion']['INTEGERS']}
</DATA INFORMATION>
Please flexibly develop a reasonable Mixed Integer Linear Programming problem based on the above DATA INFORMATION. Prohibit the use of data other than the above-mentioned information! 
<<<Reminder requirement:
1. Create a background story based on real life.
2. On the premise of ensuring that students can understand correctly, describe the content in detail, using language that is easy to understand and avoids any ambiguity.
3. Prohibit the use of any Markdown format symbols.
4. The question should include a real background, a detailed description of the data involved, specific questions. Do not include any variables! Prohibit explicitly mentioning that this is a Mixed Integer Linear Programming problem! Prohibit explicitly mentioning the use of Mixed Integer Linear Programming problem related methods for solving! Prohibit adding new constraints! Please note that BINARIES means that variables can only be equal to 0 or 1! Please strictly follow the information provided by CONSTRAINTS, especially for constraints on individual variables! INTEGERS related information must be reflected! At the same time, it must be considered that other variables are not integers!
5. Regarding the resources in the question, clear units must be provided, and "units" or "unit" should not be used in a vague manner.
6. Use <question> and </question> to package the problem description.
>>>
# RESPONSE #
A brief Mixed Integer Linear Programming problem that is easy to understand. Use <question> and </question> to package the problem. And there is no other unrelated content.
"""
    completion = client.chat.completions.create(
  model="gemini-2.0-flash",
  messages=[
    # {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
  ],
  temperature=0.9,
)
    return completion.choices[0].message.content


def gen_milp_problem(num_variables=3, num_constraints=2, int_coeff=None, coeff_range=(0, 10)):
    # 生成随机系数
    def generate_coeff():
        if int_coeff:
            return random.randint(*coeff_range)+1  # 使用 *coeff_range 解包元组作为参数
        else:
            return round(random.uniform(*coeff_range), 2)
    tool = 'Solve MILP'
    # 每个参数是否要求为整数
    if int_coeff is None:
        int_coeff = random.choice([True, False])
    # 生成决策变量名
    variables = [f'x{i + 1}' for i in range(num_variables)]
    # 生成目标函数
    objective_coeffs = [abs(generate_coeff()) for _ in range(num_variables)]
    objective = ' + '.join([f'{c}*{v}' if c != 1 else v for c, v in zip(objective_coeffs, variables)])
    # 变量非负约束
    bounds = []
    for v in variables:
        bounds.append(f'{v} >= 0')
    # 确定目标最大化还是最小化
    goal = random.choice(['Minimize', 'Maximize'])
    
    # 所有的约束
    constraints = []
    # 整数变量
    integer_variables = []
    # 二元变量
    binary_variables = []
    
    # 生成资源约束
    def gen_resource_cons(num_constraints=num_constraints):
        # 目标最大化则<= 目标最小化则>=
        constraint_sign = '<=' if goal == 'Maximize' else '>='
        num = random.randint(1, num_constraints)
        num_constraints = num_constraints - num
        for _ in range(num):
            if len(constraints)<3:
                temp_var = copy.deepcopy(variables)
            else:
                temp_var = sorted(random.sample(variables, k=random.randint(len(variables)-2, len(variables))))
            constraint_coeffs = [generate_coeff() for _ in range(len(temp_var))]
            rhs = int(generate_coeff() * random.randint(10,10000))
            constraint = ' + '.join([f'{c}*{v}' if c != 1 else v for c, v in zip(constraint_coeffs, temp_var) if c != 0])
            constraint += f' {constraint_sign} {rhs} (Resource Constraints)'
            constraints.append(constraint)
    # 生成需求约束
    def gen_demand_cons(num_constraints=num_constraints):
        if len(constraints)<num_constraints:
            var = random.choice(variables)
            constraint_sign = random.choice(['<=', '>='])
            rhs = random.randint(int(coeff_range[1]**0.5),coeff_range[1]*random.randint(1,10))
            constraint = f'{var} {constraint_sign} {rhs} (Demand Constraints)'
            constraints.append(constraint)
    # 生成策略或规则约束
    def gen_policy_regulatory_cons(num_constraints=num_constraints):
        if len(constraints)<num_constraints:
            var = sorted(random.sample(variables, k=random.randint(1, len(variables))))
            constraint_sign = random.choice(['<=', '>='])
            constraint_coeffs = [generate_coeff() for _ in range(len(var))]
            rhs = int(generate_coeff() * random.randint(10,10000))
            constraint = ' + '.join([f'{c}*{v}' if c != 1 else v for c, v in zip(constraint_coeffs, var) if c != 0])
            constraint += f' {constraint_sign} {rhs} (Policy or Regulatory Constraints)'
            constraints.append(constraint)
    # 生成平衡约束
    def gen_balance_cons(num_constraints=num_constraints):
        if len(constraints)<num_constraints:
            var = random.sample(variables, k=random.randint(2, len(variables)))
            constraint_sign = random.choice(['='])
            constraint_coeffs = [random.randint(1,10) for _ in range(len(var))]
            rhs = int(generate_coeff() * random.randint(10,10000))
            constraint = ' + '.join([f'{c}*{v}' if c != 1 else v for c, v in zip(constraint_coeffs, var) if c != 0])
            constraint += f' {constraint_sign} {rhs} (Balance Constraints)'
            constraints.append(constraint)
    # 生成比例约束
    def gen_proportionality_cons(num_constraints=num_constraints):
        if len(constraints)<num_constraints:
            # 部分变量之间成比例
            if random.random() < 0.5:
                var = random.sample(variables, k=random.randint(2, 3))
                constraint_sign = random.choice(['<=', '>=', '='])
                temp1 = 0.5 * random.randint(1,10)
                temp2 = 0.5 * random.randint(6,10) if temp1 == 2.5 else 5 - temp1
                constraint_coeffs = [temp1] + [temp2 for _ in range(len(var)-1)]
                rhs = 0
                constraint = ' - '.join([f'{c}*{v}' if c != 1 else v for c, v in zip(constraint_coeffs, var) if c != 0]).strip('-')
            # 某变量占全部产品的比例
            else:
                constraint_sign = random.choice(['<=', '>=', '='])
                temp = 0.05 * random.randint(2,19) if random.random() < 0.5 else random.randint(1,9)
                temp_vars = copy.deepcopy(variables)
                random.shuffle(temp_vars)
                if temp > 1:
                    constraint_coeffs = [10 - temp] + [temp for i in range(num_variables-1)]
                else:
                    constraint_coeffs = [round(1 - temp,2)] + [round(temp,2) for i in range(num_variables-1)]
                rhs = 0
                constraint = ' - '.join([f'{c}*{v}' if c != 1 else v for c, v in zip(constraint_coeffs, temp_vars) if c != 0]).strip('-')
            constraint += f' {constraint_sign} {rhs} (Proportionality Constraints)'
            constraints.append(constraint)
    # 逻辑约束
    def gen_logical_cons(num_constraints=num_constraints):
        if len(constraints)<num_constraints and num_variables>=3:
            constraint_type = random.choice(["if_then", "or", "not"])# "and",
            x1, x2 = random.sample(variables, k=2)
            # 整数变量中删除x1,x2
            # integer_variables.remove(x1)
            # integer_variables.remove(x2)
            # 二元变量中添加x1,x2
            binary_variables.append(x1)
            binary_variables.append(x2)
            if constraint_type == "if_then":
                # if x1 then x2  (x2 >= x1 或者 x1 - x2 <= 0)
                # if x1 then not x2 (x1 + x2 <= 1)
                if_type = random.choice([">=", "<="])
                if if_type == ">=":
                    # x2 >= x1  形式为 x1 - x2 <= 0
                    constraints.append(f"{x1} - {x2} <= 0 (Logical Constraints)") 
                else:
                    # x1 + x2 <= 1
                    constraints.append(f"{x1} + {x2} <= 1 (Logical Constraints)") 
            elif constraint_type == "or":
                # x1 or x2 (x1 + x2 >= 1)
                constraints.append(f"{x1} + {x2} >= 1 (Logical Constraints)") 
            else :# constraint_type == "not"
                # not x1 (x2 = 1 - x1), 我们生成 x1 + x2 = 1
                constraints.append(f"{x1} + {x2} = 1 (Logical Constraints)") 

    # 生成约束
    # 资源约束
    gen_resource_cons(num_constraints)
    candi_cons = [gen_demand_cons, gen_policy_regulatory_cons, gen_proportionality_cons,gen_balance_cons,gen_logical_cons]# gen_logical_cons适合IP MILP
    # 打乱顺序
    random.shuffle(candi_cons)
    for fn in candi_cons:
        fn()    
    if num_variables - len(binary_variables)>2:
        integer_variables = random.sample([v for v in variables if v not in binary_variables],random.randint(1,num_variables - len(binary_variables)))
    return {
        'forQuestion':{
            'GOAL': goal,
            'OBJECTIVE FUNCTION': objective,
            'CONSTRAINTS': constraints,
            "BINARIES": binary_variables,
            "INTEGERS":integer_variables

        },
        'forTool':{
            'tool':tool,
            'input':{
                'goal':goal,
                'objective_function': 'obj: ' + objective.replace('*', ' '),
                'constraints': [re.sub(r"\(.*?\)", "", f'c{c + 1}: {constraints[c].replace("*", " ")}').strip(' ') for c in range(len(constraints))],
                'variable_bounds': bounds,
                'variable_binaries': binary_variables,
                'variable_integers': integer_variables,
                'temp':[goal,'obj: ' + objective.replace('*', ' '),[re.sub(r"\(.*?\)", "", f'c{c + 1}: {constraints[c].replace("*", " ")}').strip(' ') for c in range(len(constraints))],bounds,binary_variables,integer_variables]
            }
        }
    }
# print(gen_lp_problem(num_constraints=5,num_variables=4))
def get_a(question):
    pattern = r"<question>(.*?)</question>"
    q = re.search(pattern, question, re.DOTALL)
    if random.random() < 0.5:
        ques =  q.group(1).strip("\n").strip().replace("\n\n", "\n")
    else:
        ques =  q.group(1).strip("\n").strip().replace("\n\n", "")
    # print(ques)
    tool_json = [{'name': 'Solve IP', 'description': 'useful when you need to solve an Integer Programming (IP: all decision variables are required to take integer values, and both the objective function and constraints are linear.) problem defined by an objective function, constraints, variable bounds, and binary variables.', 'parameters': {'type': 'object', 'properties': {'goal': {'type': 'string', 'description': "the goal of the IP, which can be either 'Maximize' or 'Minimize'.", 'enum': ['Maximize', 'Minimize']}, 'objective_function': {'type': 'string', 'description': "the objective function expressed in lp file format (e.g., 'obj: 8 x1 + 2 x2 + 3 x3')."}, 'constraints': {'type': 'array', 'description': "a list of constraints expressed in lp file format (e.g., ['c1: 4 x1 + 3 x2 <= 28', 'c2: 3 x1 + 2 x2 <= 19', 'c3: x2 + 3 x3 <= 14']).", 'items': {'type': 'string'}}, 'variable_bounds': {'type': 'array', 'description': "a list of variable bounds expressed in lp file format (e.g., ['x1 >= 0', 'x2 >= 0']).", 'items': {'type': 'string'}}, 'variable_binaries': {'type': 'array', 'description': "a list of binary variables (e.g., ['x3']).", 'items': {'type': 'string'}}}, 'required': ['goal', 'objective_function', 'constraints', 'variable_bounds', 'variable_binaries']}}, {'name': 'Solve LP', 'description': 'useful when you need to solve a Linear Programming (LP: find the best outcome in a model with linear relationships, subject to a set of constraints.) problem defined by an objective function, constraints, and variable bounds.', 'parameters': {'type': 'object', 'properties': {'goal': {'type': 'string', 'description': "the goal of the LP, which can be either 'Maximize' or 'Minimize'.", 'enum': ['Maximize', 'Minimize']}, 'objective_function': {'type': 'string', 'description': "the objective function expressed in lp file format (e.g., 'obj: 8 x + 15 y')."}, 'constraints': {'type': 'array', 'description': "a list of constraints expressed in lp file format (e.g., ['c1: 10 x + 15 y <= 3000', 'c2: x + y <= 250']).", 'items': {'type': 'string'}}, 'variable_bounds': {'type': 'array', 'description': "a list of variable bounds expressed in lp file format (e.g., ['x >= 0', 'y >= 0']).", 'items': {'type': 'string'}}}, 'required': ['goal', 'objective_function', 'constraints', 'variable_bounds']}}, {'name': 'Solve MILP', 'description': 'useful when you need to solve a Mixed Integer Linear Programming (MILP: the objective function and constraints are linear, but some of the decision variables are restricted to integer values, while others can be continuous.) problem defined by an objective function, constraints, variable bounds, binary variables, and integer variables.', 'parameters': {'type': 'object', 'properties': {'goal': {'type': 'string', 'description': "the goal of the MILP, which can be either 'Maximize' or 'Minimize'.", 'enum': ['Maximize', 'Minimize']}, 'objective_function': {'type': 'string', 'description': "the objective function expressed in lp file format (e.g., 'obj: 3 x1 + 5 x2 + x3 + x4')."}, 'constraints': {'type': 'array', 'description': "a list of constraints expressed in lp file format (e.g., ['c1: 2 x1 + x2 + x4 <= 18.5', 'c2: x1 + 2 x2 <= 15', 'c3: x2 + x3 <= 8.5']).", 'items': {'type': 'string'}}, 'variable_bounds': {'type': 'array', 'description': "a list of variable bounds expressed in lp file format (e.g., ['x1 >= 0', 'x2 >= 0', 'x3 >= 0']).", 'items': {'type': 'string'}}, 'variable_binaries': {'type': 'array', 'description': "a list of binary variables (e.g., ['x4']).", 'items': {'type': 'string'}}, 'variable_integers': {'type': 'array', 'description': "a list of integer variables (e.g., ['x1', 'x2']).", 'items': {'type': 'string'}}}, 'required': ['goal', 'objective_function', 'constraints', 'variable_bounds', 'variable_binaries', 'variable_integers']}}]
    system = f"""You are a mathematical modeling expert and a professor of operations research at a top university. You are very good at solving various operations research problems. When solving operations research problems, you will first conduct mathematical modeling (if necessary), and then use appropriate tools to obtain the optimal solution. Users can provide you with problem descriptions in plain text, but any problem must be solved using the correct tools. 

<tools>{tool_json}</tools>
When you need to use a tool, please follow this format: <use_tool>tool_name(param1, param2, ...)</use_tool>

Always use the data provided in the problem description to avoid fabricating any incorrect data. Remember to pass in accurate and complete parameters to the tool without omission!"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{ques}\n\nYou need to use a suitable tool to solve this problem, rather than solving it directly. Let’s think step by step. Prohibit including JSON format content!"}]
    completion = client.chat.completions.create(
  model="gemini-2.0-flash",
  messages=messages,
  temperature=0,
)
    return completion.choices[0].message.content
    
def judge(data,answer,label_output):
    label_tool = data['forTool']['tool']
    
    pred_tool = extract_tool_name(answer)
    
    # print(label_tool)
    # print(pred_tool)
    # print(label_input)
    # print(pred_input)
    if label_tool == pred_tool:
        label_input = data['forTool']['input']['temp']
        pred_input = extract_tool_input(answer)
        pred_input = eval(f'extract_tool_milp_input({pred_input})')
        if len(pred_input)!= len(label_input):

            return False
        else:
            if label_input==pred_input:
                print("二者相同：",label_input, pred_input)

                return True         
            else:
                print("执行函数进行判断。。。")
            # 函数调用
                # label_output = eval(f'{TOOL2FUNCNAME[label_tool]}(*{label_input})') 
                pred_output = eval(f'{TOOL2FUNCNAME[pred_tool]}(*{pred_input})')
                print('label_output:',label_output)
                print('pred_output:',pred_output)
                if label_output[0]!=pred_output[0] or label_output[1]!=pred_output[1]:
                    return False
                return True
    else:
        return False


if __name__ == '__main__':
    client = OpenAI(
    base_url='YOUR_BASE_URL',
    api_key='YOUR_API_KEY')
    result = {
    'system':[],
    'user':[],
    'assistant':[],
    'tool':[],
    'input':[],
    'output':[],
    'type':[],
    'question_params':[],
    'pred_input':[],
    # 'pred_output':[],
    'judge':[]

}
    work = Queue()
    for i in range(30000): #
        work.put_nowait(i)
    def worker():
        while not work.empty():
            current = work.get()
            if len(result['system'])>=728:###########
                # 清空任务队列
                while not work.empty():
                    work.get()
                break
            try:
                data = gen_milp_problem(num_variables=random.randint(2,5),num_constraints=random.randint(2,5),coeff_range=(10,1000))
                # TODO 如果正确的输出就是None，就不保留
                label_tool = data['forTool']['tool']
                label_input = data['forTool']['input']['temp']
                label_output = eval(f'{TOOL2FUNCNAME[label_tool]}(*{label_input})')
                if label_output[0] is None or label_output=='[None, None]':
                    print(f'正确答案为{label_output[0]}，舍去')
                    continue
                question = get_q(data)
                answer = get_a(question)
                judge_result = judge(data,answer,label_output)
                pred_input = extract_tool_input(answer)
                if judge_result:
                    pattern = r"<question>(.*?)</question>"
                    q = re.search(pattern, question, re.DOTALL)
                    if random.random() < 0.5:
                        ques =  q.group(1).strip("\n").strip().replace("\n\n", "\n")
                    else:
                        ques =  q.group(1).strip("\n").strip().replace("\n\n", "")
                    ans = answer.replace('```tool_code','').replace('```','').replace("\n\n\n","\n\n").strip('\n')
                    necessary_tools = []
                    necessary_tools.append(data['forTool']['tool'])
                    candidate_tools = list(TOOLS_DESCRIPTION_JSON.keys())
                    # Tool Name List
                    x = y = 4
                    tools_name_list = select_tools(necessary_tools, candidate_tools, x, y)
                    # Tools Description List
                    tools = [TOOLS_DESCRIPTION_JSON[tool] for tool in tools_name_list]
                    system = f"""You are a mathematical modeling expert and a professor of operations research at a top university. You are very good at solving various operations research problems. When solving operations research problems, you will first conduct mathematical modeling (if necessary), and then use appropriate tools to obtain the optimal solution. Users can provide you with problem descriptions in plain text, but any problem must be solved using the correct tools. 
                    
<tools>{tools}</tools>

When you need to use a tool, please follow this format: <use_tool>tool_name(param1, param2, ...)</use_tool>

Always use the data provided in the problem description to avoid fabricating any incorrect data. Remember to pass in accurate and complete parameters to the tool without omission!"""
                    user = ques + "\n\nYou need to use a suitable tool to solve this problem, rather than solving it directly. Let’s think step by step."
                    assistant = ans
                    # assistant = clear_assistant(ans)

                    result['system'].append(system)
                    result['user'].append(user)
                    result['assistant'].append(assistant)
                    result['tool'].append(data['forTool']['tool'])
                    result['input'].append(data['forTool']['input']['temp'])
                    result['output'].append(label_output)
                    # ==================================================
                    result['type'].append('MILP')
                    # ==================================================
                    result['question_params'].append(data['forQuestion'])
                    result['pred_input'].append(pred_input)
                    # result['pred_output'].append(answer)
                    result['judge'].append(judge_result)
                    print('已生成数据：',len(result['system']))
                    result_df = pd.DataFrame(result)
                    result_df.to_excel('/home/bigchill/mypipe/genChat5/MILP/MILP_Chat5.xlsx', index=False)
                else:
                    continue
            except Exception as e:
                print('error:',e)
                # if 'error' in str(e):
                #     time.sleep(5)
    tasks_list = []
    for x in range(15):
        task = gevent.spawn(worker)
        tasks_list.append(task)
    gevent.joinall(tasks_list)
