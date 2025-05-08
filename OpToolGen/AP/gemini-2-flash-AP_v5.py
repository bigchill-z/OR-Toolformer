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
    prompt = f"""# CONTEXT #
You are a professional operations research teacher, skilled in designing problems related to operations research. To help students further master Assignment Problem, we now need you to set a reasonable Assignment Problem.
# OBJECTIVE #
<DATA INFORMATION>
INDUSTRY: {random.choice(['Energy', 'Health', 'Retail', 'Environment', 'Education', 'Agriculture', 'Public Utilities', 'Manufacturing', 'Software', 'Construction', 'Entertainment Legal', 'Customer Service', 'Transportation', 'Financial Services'])}
GOAL: {data['forQuestion']['TYPE']},
NUM AGENTS: {data['forQuestion']['NUM_AGENTS']},
NUM TASKS: {data['forQuestion']['NUM_TASKS']},
DATA TYPE: {data['forQuestion']['DATA TYPE']},
SPECIFIC DATA: {data['forQuestion']['SPECIFIC DATA']}
</DATA INFORMATION>
Please flexibly develop a reasonable Assignment Problem based on the above DATA INFORMATION. Prohibit the use of data other than the above-mentioned information! 
<<<Reminder requirement:
1. Create a background story based on real life.
2. On the premise of ensuring that students can understand correctly, describe the content in detail, using language that is easy to understand and avoids any ambiguity.
3. Prohibit the use of any Markdown format symbols.
4. The question should include a real background, a detailed description of the data involved, specific questions. Prohibit explicitly mentioning that this is a Assignment Problem! Prohibit explicitly mentioning the use of Assignment Problem related methods for solving!
5. Regarding the resources in the question, clear units must be provided, and "units" or "unit" should not be used in a vague manner.
6. Use <question> and </question> to package the problem description.
>>>
# RESPONSE #
A brief Assignment Problem that is easy to understand. Use <question> and </question> to package the problem description. And there is no other unrelated content.
"""
    completion = client.chat.completions.create(
  model="gemini-2.0-flash",
  messages=[
    {"role": "user", "content": prompt}
  ],
  temperature=0.9,
)
    return completion.choices[0].message.content
def generate_ap_list(num_agents=4,num_tasks=3, mean=0, variance=1):
    """

  参数:
      num_agents (int): 矩阵的行数。
      num_tasks (int): 矩阵的列数。
      mean (float): 正态分布的均值。
      variance (float): 正态分布的方差。

  """
        # 生成符合正态分布的随机数
    matrix = np.random.normal(loc=mean, scale=variance, size=(num_agents, num_tasks))

    # 将随机数转换为整数
    matrix = np.round(matrix).astype(int).tolist()
    agents = []
    tasks = []
    weights = []
    goal = random.choice(['Maximize', 'Minimize'])
    table = """"""
    for i in range(num_agents):
        for j in range(num_tasks):
            if goal == 'Minimize':
                table += f'AGENT{i + 1} assigned to TASK{j + 1}. Cost = {matrix[i][j]}\n'
            else:
                table += f'AGENT{i + 1} assigned to TASK{j + 1}. Benefit = {matrix[i][j]}\n'
            agents.append(f'AGENT{i + 1}')
            tasks.append(f'TASK{j + 1}')
            weights.append(f'{matrix[i][j]}')
    # print(table)

    tool = 'Solve AP with List'
    return {
        # 用于替换问题模板的内容
        'forQuestion': {
            'TYPE': goal,
            'NUM_AGENTS': num_agents,
            'NUM_TASKS': num_tasks,
            'DATA TYPE': 'list',
            'SPECIFIC DATA': table,
        },
        # 用于传入API调用的参数内容
        'forTool': {
            'tool': tool,
            'input': {
                'goal': goal,
                'agents': agents,
                'tasks': tasks,
                'weights': weights,
                'temp': [goal, agents, tasks, weights]             
            }
    }
}
# 示例调用
# data = generate_ap_list(num_agents=4,num_tasks=3, mean=100, variance=25)
# print(json.dumps(data, indent=4))

def generate_ap_matrix(num_agents=4,num_tasks=3, mean=0, variance=1):
    """
  生成一个符合正态分布的随机整数矩阵。

  参数:
      num_agents (int): 矩阵的行数。
      num_tasks (int): 矩阵的列数。
      mean (float): 正态分布的均值。
      variance (float): 正态分布的方差。

  返回值:
      numpy.ndarray: 生成的随机数矩阵。
  """
    # 生成符合正态分布的随机数
    matrix = np.random.normal(loc=mean, scale=variance, size=(num_agents, num_tasks))

    # 将随机数转换为整数
    matrix = np.round(matrix).astype(int).tolist()
    goal = random.choice(['Maximize', 'Minimize'])
    tool = 'Solve AP with Matrix'
    return {
        # 用于替换问题模板的内容
        'forQuestion': {
            'TYPE': goal,
            'NUM_AGENTS': num_agents,
            'NUM_TASKS': num_tasks,
            'DATA TYPE': 'matrix',
            'SPECIFIC DATA': matrix,
        },
        # 用于传入API调用的参数内容
        'forTool': {
            'tool': tool,
            'input': {
                'goal': goal,
                'matrix_data': matrix,
                'temp': [goal, matrix]
            }
    }
}

# 示例调用, 
# data = generate_ap_matrix(num_agents=4,num_tasks=3, mean=100, variance=25)
# print(json.dumps(data,indent=4))
def get_a(question):
    pattern = r"<question>(.*?)</question>"
    q = re.search(pattern, question, re.DOTALL)
    if random.random() < 0.5:
        ques =  q.group(1).strip("\n").strip().replace("\n\n", "\n")
    else:
        ques =  q.group(1).strip("\n").strip().replace("\n\n", "")
    tool_json = [
                {'name': 'Solve IP', 'description': 'useful when you need to solve an Integer Programming (IP: all decision variables are required to take integer values, and both the objective function and constraints are linear.) problem defined by an objective function, constraints, variable bounds, and binary variables.', 'parameters': {'type': 'object', 'properties': {'goal': {'type': 'string', 'description': "the goal of the IP, which can be either 'Maximize' or 'Minimize'.", 'enum': ['Maximize', 'Minimize']}, 'objective_function': {'type': 'string', 'description': "the objective function expressed in lp file format (e.g., 'obj: 8 x1 + 2 x2 + 3 x3')."}, 'constraints': {'type': 'array', 'description': "a list of constraints expressed in lp file format (e.g., ['c1: 4 x1 + 3 x2 <= 28', 'c2: 3 x1 + 2 x2 <= 19', 'c3: x2 + 3 x3 <= 14']).", 'items': {'type': 'string'}}, 'variable_bounds': {'type': 'array', 'description': "a list of variable bounds expressed in lp file format (e.g., ['x1 >= 0', 'x2 >= 0']).", 'items': {'type': 'string'}}, 'variable_binaries': {'type': 'array', 'description': "a list of binary variables (e.g., ['x3']).", 'items': {'type': 'string'}}}, 'required': ['goal', 'objective_function', 'constraints', 'variable_bounds', 'variable_binaries']}}, 
                {'name': 'Solve AP with List', 'description': 'useful when you need to solve an Assignment Problem (AP: the goal is to assign a set of tasks to a set of agents in such a way that minimizes the total cost or maximizes the total profit, subject to the limitation that each task is assigned to exactly one agent and each agent is assigned exactly one task.) using lists of agents, tasks, and their associated weights.', 'parameters': {'type': 'object', 'properties': {'goal': {'type': 'string', 'description': "the goal of the assignment problem, which can be either 'Maximize' or 'Minimize'.", 'enum': ['Maximize', 'Minimize']}, 'agents': {'type': 'array', 'description': "a list of agent identifiers (e.g., ['Agent1', 'Agent1', 'Agent1', 'Agent1', 'Agent2', ...]).", 'items': {'type': 'string'}}, 'tasks': {'type': 'array', 'description': "a list of task identifiers (e.g., ['Task1', 'Task2', 'Task3', 'Task4', 'Task1', ...]).", 'items': {'type': 'string'}}, 'weights': {'type': 'array', 'description': 'a list of weights corresponding to the assignment costs or benefits (e.g., [132, 103, 100, 56, 96, ...]).', 'items': {'type': 'number'}}}, 'required': ['goal', 'agents', 'tasks', 'weights']}},
                {'name': 'Solve AP with Matrix', 'description': 'useful when you need to solve an Assignment Problem (AP: the goal is to assign a set of tasks to a set of agents in such a way that minimizes the total cost or maximizes the total profit, subject to the limitation that each task is assigned to exactly one agent and each agent is assigned exactly one task.) using a matrix representation of costs or benefits.', 'parameters': {'type': 'object', 'properties': {'goal': {'type': 'string', 'description': "the goal of the assignment problem, which can be either 'Maximize' or 'Minimize'.", 'enum': ['Maximize', 'Minimize']}, 'matrix_data': {'type': 'array', 'description': 'a 2D array where each element represents the cost (for minimization) or benefit (for maximization) of assigning agent i to task j.', 'items': {'type': 'array', 'items': {'type': 'number'}}}, 'agents': {'type': 'array', 'description': "a list of agent identifiers (e.g., ['w1', 'w2', 'w3', 'w4', 'w5']).", 'items': {'type': 'string'}}, 'tasks': {'type': 'array', 'description': "a list of task identifiers (e.g., ['t1', 't2', 't3', 't4']).", 'items': {'type': 'string'}}}, 'required': ['goal', 'matrix_data']}}
                ]
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
        pred_input = eval(f'extract_tool_ap_input(("{pred_tool}","{pred_input}"))')
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
                if label_output[0]!=pred_output[0]:
                    return False
                return True
    else:
        return False

if __name__ == '__main__':
    client = OpenAI(
    base_url='YOUR_BASE_URL',
    api_key='YOUR_API_KEY'
)   

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
    for i in range(200000): #
        work.put_nowait(i)
    def worker():
        while not work.empty():
            current = work.get()
            print(current)
            if len(result['system'])>=100:
                # 清空任务队列
                while not work.empty():
                    work.get()
                break
            try:
                num_tasks = random.randint(4,5)
                num_agents = num_tasks + random.randint(0,1)
                mean = random.randint(1,20)*100
                var = int(mean/4)
                if random.random() < 0:
                    data = generate_ap_list(num_agents=num_agents,num_tasks=num_tasks,mean=mean,variance=var)
                else:
                    data = generate_ap_matrix(num_agents=num_agents,num_tasks=num_tasks,mean=mean,variance=var)
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
                    print('judge_result:',judge_result)
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
                    #############################################################################################################################################
                    TOOLS_DESCRIPTION_ALL_JSON = TOOLS_DESCRIPTION_JSON | TOOLS_DESCRIPTION_PLUS_JSON
                    tools = [TOOLS_DESCRIPTION_ALL_JSON[tool] for tool in tools_name_list]
                    #############################################################################################################################################
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
                    result['type'].append('AP')
                    # ==================================================
                    result['question_params'].append(data['forQuestion'])
                    result['pred_input'].append(pred_input)
                    # result['pred_output'].append(answer)
                    result['judge'].append(judge_result)
                    print('已生成数据：',len(result['system']))
                    result_df = pd.DataFrame(result)
                    result_df.to_excel('/home/bigchill/mypipe/genChat5/AP/AP_Chat5.xlsx', index=False)
                else:
                    print('未生成数据')
                    continue
            except Exception as e:
                print('error:',e)
                # if 'error' in str(e):
                #     time.sleep(random.random())
    tasks_list = []
    for x in range(5):
        task = gevent.spawn(worker)
        tasks_list.append(task)
    gevent.joinall(tasks_list)
