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
You are a professional operations research teacher, skilled in designing problems related to operations research. To help students further master Maximum Flow, we now need you to set a reasonable Maximum Flow.
# OBJECTIVE #
<DATA INFORMATION>
INDUSTRY: {random.choice(['Energy', 'Health', 'Retail', 'Environment', 'Education', 'Agriculture', 'Public Utilities', 'Manufacturing', 'Software', 'Construction', 'Entertainment Legal', 'Customer Service', 'Transportation', 'Financial Services'])}
DATA TYPE: {data['forQuestion']['DATA TYPE']},
SPECIFIC DATA: {data['forQuestion']['SPECIFIC DATA']}
</DATA INFORMATION>
Please flexibly develop a reasonable Maximum Flow based on the above DATA INFORMATION. Prohibit the use of data other than the above-mentioned information! 
<<<Reminder requirement:
1. Create a background story based on real life.
2. On the premise of ensuring that students can understand correctly, describe the content in detail, using language that is easy to understand and avoids any ambiguity.
3. Prohibit the use of any Markdown format symbols.
4. The question should include a real background, a detailed description of the data involved, specific questions. Prohibit explicitly mentioning that this is a Maximum Flow! Prohibit explicitly mentioning the use of Maximum Flow related methods for solving!
5. Regarding the resources in the question, clear units must be provided, and "units" or "unit" should not be used in a vague manner.
6. Use <question> and </question> to package the problem description.
>>>
# RESPONSE #
A brief Maximum Flow that is easy to understand. Use <question> and </question> to package the problem. And there is no other unrelated content.
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

def generate_max_flow_matrix(num_nodes, min_capacity=1, max_capacity=10, sparsity=0.5):
    """
    生成一个随机的网络流矩阵，用于最大流问题。

    参数:
    num_nodes: 节点数量 (包括源节点和汇节点).
    min_capacity: 边容量的最小值.
    max_capacity: 边容量的最大值.
    sparsity: 矩阵的稀疏度 (0 到 1 之间的数值). 较小的值表示更稀疏的矩阵.

    返回值:
    numpy 数组: 表示网络流图的容量矩阵.
    """

    # 初始化容量矩阵，所有边容量初始为0
    capacity_matrix = np.zeros((num_nodes, num_nodes))

    # 随机生成边和容量
    for i in range(num_nodes):
        for j in range(num_nodes):
            # 跳过自身连接
            if i == j:
                continue
            # 根据稀疏度随机生成边
            if np.random.rand() < sparsity:
                capacity_matrix[i, j] = np.random.randint(min_capacity, max_capacity + 1)
    # 确保每个点一定有出边
    for i in range(num_nodes):
        if capacity_matrix[i, :].sum() == 0:
            capacity_matrix[i, np.random.randint(i+1, num_nodes)] = np.random.randint(min_capacity, max_capacity + 1)

    # 确保源节点只有出边，汇节点只有入边
    source = 0
    sink = num_nodes - 1
    capacity_matrix[:, source] = 0  # 源节点没有入边
    capacity_matrix[sink, :] = 0  # 汇节点没有出边

    data = capacity_matrix.astype(int).tolist()
    tool = 'Solve MF with Matrix'
    return {
        # 用于替换问题模板的内容
        'forQuestion': {
            'DATA TYPE': 'matrix',
            'SPECIFIC DATA': data,
        },
        # 用于传入API调用的参数内容
        'forTool': {
            'tool': tool,
            'input': {
                'matrix': data,
                'temp': [data],
            }
        }
    } 

# 示例用法:
# data = generate_max_flow_matrix(num_nodes=6,min_capacity=10,max_capacity=60,sparsity=0.7)
# print(json.dumps(data, indent=4))
# data

def generate_max_flow_list(num_nodes, min_capacity=1, max_capacity=10, sparsity=0.5):
    matrix = generate_max_flow_matrix(num_nodes,min_capacity,max_capacity,sparsity)
    matrix = matrix['forQuestion']['SPECIFIC DATA']
    start_nodes = []
    end_nodes = []
    capacities = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] != 0:  # 假设非零值表示有边
                start_nodes.append(i)
                end_nodes.append(j)
                capacities.append(int(matrix[i][j]))

    data = {'start_nodes': start_nodes, 'end_nodes': end_nodes, 'capacities': capacities}
    tool = 'Solve MF with List'
    return {
        # 用于替换问题模板的内容
        'forQuestion': {
            'DATA TYPE': 'list',
            'SPECIFIC DATA':random.choice([[data],[f'{s}->{e}: {c}' for s, e ,c in zip(data['start_nodes'],data['end_nodes'],data['capacities'])]]),
        },
        # 用于传入API调用的参数内容
        'forTool': {
            'tool': tool,
            'input': {
                'start_nodes': data['start_nodes'],
                'end_nodes': data['end_nodes'],
                'capacities': data['capacities'],
                'temp': [data['start_nodes'], data['end_nodes'], data['capacities']],
            },
        }
    } 
# # 示例用法:
# data = generate_max_flow_list(6)
# print(json.dumps(data,indent=8))

def get_a(question):
    pattern = r"<question>(.*?)</question>"
    q = re.search(pattern, question, re.DOTALL)
    if random.random() < 0.5:
        ques =  q.group(1).strip("\n").strip().replace("\n\n", "\n")
    else:
        ques =  q.group(1).strip("\n").strip().replace("\n\n", "")
    tool_json = [{'name': 'Solve IP', 'description': 'useful when you need to solve an Integer Programming (IP: all decision variables are required to take integer values, and both the objective function and constraints are linear.) problem defined by an objective function, constraints, variable bounds, and binary variables.', 'parameters': {'type': 'object', 'properties': {'goal': {'type': 'string', 'description': "the goal of the IP, which can be either 'Maximize' or 'Minimize'.", 'enum': ['Maximize', 'Minimize']}, 'objective_function': {'type': 'string', 'description': "the objective function expressed in lp file format (e.g., 'obj: 8 x1 + 2 x2 + 3 x3')."}, 'constraints': {'type': 'array', 'description': "a list of constraints expressed in lp file format (e.g., ['c1: 4 x1 + 3 x2 <= 28', 'c2: 3 x1 + 2 x2 <= 19', 'c3: x2 + 3 x3 <= 14']).", 'items': {'type': 'string'}}, 'variable_bounds': {'type': 'array', 'description': "a list of variable bounds expressed in lp file format (e.g., ['x1 >= 0', 'x2 >= 0']).", 'items': {'type': 'string'}}, 'variable_binaries': {'type': 'array', 'description': "a list of binary variables (e.g., ['x3']).", 'items': {'type': 'string'}}}, 'required': ['goal', 'objective_function', 'constraints', 'variable_bounds', 'variable_binaries']}}, 
                 {'name': 'Solve MF with Matrix', 'description': 'useful when you need to solve a Maximum Flow (MF: find the greatest possible flow of resources from a source node to a sink node in a network, while respecting the capacities of the edges.) problem using a matrix representation of the capacities.', 'parameters': {'type': 'object', 'properties': {'matrix': {'type': 'array', 'description': 'a 2D array where non-zero values represent the capacity of the edges between nodes. The element at position [i][j] indicates the capacity from node i to node j.', 'items': {'type': 'array', 'items': {'type': 'integer'}}}}, 'required': ['matrix']}},
                 {'name': 'Solve MF with List', 'description': 'useful when you need to solve a Maximum Flow (MF: find the greatest possible flow of resources from a source node to a sink node in a network, while respecting the capacities of the edges.) problem using a list representation of edges.', 'parameters': {'type': 'object', 'properties': {'start_nodes': {'type': 'array', 'description': 'a one-dimensional array representing the starting nodes of each edge.', 'items': {'type': 'integer'}}, 'end_nodes': {'type': 'array', 'description': 'a one-dimensional array representing the ending nodes of each edge.', 'items': {'type': 'integer'}}, 'capacities': {'type': 'array', 'description': 'a one-dimensional array representing the capacity limits for each edge.', 'items': {'type': 'integer'}}}, 'required': ['start_nodes', 'end_nodes', 'capacities']}}
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
    
# answer = get_a(result)
# print(answer)

def judge(data,answer,label_output):
    label_tool = data['forTool']['tool']
    
    pred_tool = extract_tool_name(answer)

    if label_tool == pred_tool:
        label_input = data['forTool']['input']['temp']
        pred_input = extract_tool_input(answer)
        # print(answer)
        pred_input = eval(f'extract_tool_mf_input(("{pred_tool}","{pred_input}"))')
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
    for i in range(10000): #
        work.put_nowait(i)
    def worker():
        while not work.empty():
            current = work.get()
            if len(result['system'])>=19:
                # 清空任务队列
                while not work.empty():
                    work.get()
                break
            try:
                num_nodes = random.randint(4,6)
                minc = random.randint(10,100)
                maxc = minc * 10
                if random.random() < 0.5:
                    data = generate_max_flow_matrix(num_nodes,minc,maxc)
                else:
                    data = generate_max_flow_list(num_nodes,minc,maxc)
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
                    result['type'].append('MF')
                    # ==================================================
                    result['question_params'].append(data['forQuestion'])
                    result['pred_input'].append(pred_input)
                    # result['pred_output'].append(answer)
                    result['judge'].append(judge_result)
                    print('已生成数据：',len(result['system']))
                    result_df = pd.DataFrame(result)
                    result_df.to_excel('/home/bigchill/mypipe/genChat5/MF/MF_Chat5_test.xlsx', index=False)
                else:
                    continue
            except Exception as e:
                # print(answer)
                print(e)
                # if 'error' in str(e):
                #     time.sleep(5)
    tasks_list = []
    for x in range(8):
        task = gevent.spawn(worker)
        tasks_list.append(task)
    gevent.joinall(tasks_list)
