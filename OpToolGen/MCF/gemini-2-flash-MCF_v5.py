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
You are a professional operations research teacher, skilled in designing problems related to operations research. To help students further master Minimum Cost Flow, we now need you to set a reasonable Minimum Cost Flow.
# OBJECTIVE #
<DATA INFORMATION>
INDUSTRY: {random.choice(['Energy', 'Health', 'Retail', 'Environment', 'Education', 'Agriculture', 'Public Utilities', 'Manufacturing', 'Software', 'Construction', 'Entertainment Legal', 'Customer Service', 'Transportation', 'Financial Services'])}
DATA TYPE: {data['forQuestion']['DATA TYPE']},
SPECIFIC DATA: {data['forQuestion']['SPECIFIC DATA']}
</DATA INFORMATION>
Please flexibly develop a reasonable Minimum Cost Flow based on the above DATA INFORMATION. Prohibit the use of data other than the above-mentioned information! 
<<<Reminder requirement:
1. Create a background story based on real life.
2. On the premise of ensuring that students can understand correctly, describe the content in detail, using language that is easy to understand and avoids any ambiguity.
3. Prohibit the use of any Markdown format symbols.
4. The question should include a real background, a detailed description of the data involved, specific questions. Please carefully identify which are supply points and which are demand points. Prohibit explicitly mentioning that this is a minimum cost flow problem! Prohibit explicitly mentioning the use of minimum cost flow problem related methods for solving!
5. Regarding the resources in the question, clear units must be provided, and "units" or "unit" should not be used in a vague manner.
6. Use <question> and </question> to package the problem description.
>>>
# RESPONSE #
A brief Minimum Cost Flow that is easy to understand. Use <question> and </question> to package the problem description. And there is no other unrelated content.
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
def generate_min_cost_flow_matrix(num_nodes, num_sources, num_sinks, min_capacity=10, max_capacity=100, min_cost=1, max_cost=20, sparsity=0.5):
    """
    生成一个随机的最小费用最大流问题的容量矩阵、成本矩阵和供应列表。

    参数:
    num_nodes: 节点数量.
    num_sources: 源节点的数量.
    num_sinks: 汇节点的数量.
    min_capacity: 边容量的最小值.
    max_capacity: 边容量的最大值.
    min_cost: 边成本的最小值.
    max_cost: 边成本的最大值.
    sparsity: 矩阵的稀疏度 (0 到 1 之间的数值). 较小的值表示更稀疏的矩阵.

    返回值:
    tuple: 包含容量矩阵、成本矩阵和供应列表.
    """
    if num_sources + num_sinks >= num_nodes:
        raise ValueError("源节点和汇节点的总数必须小于节点总数")

    # 初始化容量矩阵和成本矩阵
    capacity_matrix = np.zeros((num_nodes, num_nodes))
    cost_matrix = np.zeros((num_nodes, num_nodes))

    # 定义源节点和汇节点的索引
    sources = list(range(num_sources))  # 源节点索引
    sinks = list(range(num_nodes - num_sinks, num_nodes))  # 汇节点索引

    # 随机生成边、容量和成本
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            if np.random.rand() < sparsity:
                capacity = np.random.randint(min_capacity, max_capacity + 1)
                cost = np.random.randint(min_cost, max_cost + 1)
                capacity_matrix[i, j] = capacity
                cost_matrix[i, j] = cost

    # 确保每个点一定有出边
    for i in range(num_nodes):
        if capacity_matrix[i, :].sum() == 0:
            capacity_matrix[i, np.random.randint(i + 1, num_nodes)] = np.random.randint(min_capacity, max_capacity + 1)
            cost_matrix[i, np.random.randint(i + 1, num_nodes)] = np.random.randint(min_cost, max_cost + 1)

    # 确保源节点只有出边，汇节点只有入边
    for source in sources:
        capacity_matrix[:, source] = 0  # 源节点没有入边
        cost_matrix[:, source] = 0       # 源节点的入边成本设为0

    for sink in sinks:
        capacity_matrix[sink, :] = 0      # 汇节点没有出边
        cost_matrix[sink, :] = 0           # 汇节点没有出边的成本设为零

    # 生成供应列表
    supply = np.zeros(num_nodes)
    total_supply = np.random.randint(20, 200)  # 随机生成总供应量
    total_require = total_supply
    # 随机分配源节点的供应量
    for source in sources:
        if sources.index(source) == len(sources) - 1:
            supply[source] = total_supply
        else:
            supply[source] = np.random.randint(1, total_supply - len(sources) + 1)
            total_supply -= supply[source]
        
    for sink in sinks:
        if sinks.index(sink) == len(sinks) - 1:
            supply[sink] = -total_require
        else:
            supply[sink] = -np.random.randint(1, total_require - len(sinks) + 1)
            total_require -= -supply[sink]

    result = [capacity_matrix.astype(int).tolist(), cost_matrix.astype(int).tolist(), supply.astype(int).tolist()]

    tool = 'Solve MCF with Matrix'
    return {
        # 用于替换问题模板的内容
        'forQuestion': {
            'DATA TYPE': 'matrix', 
            'SPECIFIC DATA': {
                'capacity_matrix': result[0],
                'cost_matrix': result[1],
                'supply': result[2],
            }
        },
        # 用于传入API调用的参数内容
        'forTool': {
            'tool': tool,
            'input': {
                'capacity_matrix': result[0], 
                'cost_matrix': result[1], 
                'supplies': result[2],
                'temp': result
            },
        }
    } 

# 示例用法:
# data = generate_min_cost_flow_matrix(num_nodes=6, num_sources=2, num_sinks=2)
# print(json.dumps(data, indent=4))

def generate_min_cost_flow_list(num_nodes, num_sources, num_sinks, min_capacity=10, max_capacity=100, min_cost=1, max_cost=20):
    matrix = generate_min_cost_flow_matrix(num_nodes, num_sources, num_sinks, min_capacity, max_capacity, min_cost, max_cost)
    matrix = matrix['forQuestion']['SPECIFIC DATA']
    capacities_matrix = matrix['capacity_matrix']
    costs_matrix = matrix['cost_matrix']
    supplies = matrix['supply']
    start_nodes = []
    end_nodes = []
    capacities = []
    costs =[]
    for i in range(len(capacities_matrix)):
        for j in range(len(capacities_matrix[i])):
            if capacities_matrix[i][j] != 0 and costs_matrix[i][j] != 0:  # 假设非零值表示有边
                start_nodes.append(i)
                end_nodes.append(j)
                capacities.append(int(capacities_matrix[i][j]))
                costs.append(int(costs_matrix[i][j]))

    data = {'start_nodes': start_nodes, 'end_nodes': end_nodes, 'capacities': capacities,'unit_costs':costs}
    tool = 'Solve MCF with List'
    return {
        # 用于替换问题模板的内容
        'forQuestion': {
            'DATA TYPE': 'list',
            'SPECIFIC DATA':{
                'capacity_cost':random.choice([[data],[f'{s}->{e}. capacity: {c}. unit cost: {cost}' for s, e ,c,cost in zip(data['start_nodes'],data['end_nodes'],data['capacities'],data['unit_costs'])]]),
                'supplies':supplies
            },
        },
        # 用于传入API调用的参数内容
        'forTool': {
            'tool': tool,
            'input': {
                'start_nodes': data['start_nodes'],
                'end_nodes': data['end_nodes'],
                'capacities': data['capacities'],
                'unit_costs':data['unit_costs'],
                'supplies': supplies,
                'temp': [data['start_nodes'], data['end_nodes'], data['capacities'], data['unit_costs'], supplies],
            }
        }
    } 

# 示例用法:
# data = generate_min_cost_flow_list(num_nodes=6, num_sources=2, num_sinks=2)
# print(json.dumps(data, indent=4))

def get_a(question):
    pattern = r"<question>(.*?)</question>"
    q = re.search(pattern, question, re.DOTALL)
    if random.random() < 0.5:
        ques =  q.group(1).strip("\n").strip().replace("\n\n", "\n")
    else:
        ques =  q.group(1).strip("\n").strip().replace("\n\n", "")
    tool_json = [
                 {'name': 'Solve IP', 'description': 'useful when you need to solve an Integer Programming (IP: all decision variables are required to take integer values, and both the objective function and constraints are linear.) problem defined by an objective function, constraints, variable bounds, and binary variables.', 'parameters': {'type': 'object', 'properties': {'goal': {'type': 'string', 'description': "the goal of the IP, which can be either 'Maximize' or 'Minimize'.", 'enum': ['Maximize', 'Minimize']}, 'objective_function': {'type': 'string', 'description': "the objective function expressed in lp file format (e.g., 'obj: 8 x1 + 2 x2 + 3 x3')."}, 'constraints': {'type': 'array', 'description': "a list of constraints expressed in lp file format (e.g., ['c1: 4 x1 + 3 x2 <= 28', 'c2: 3 x1 + 2 x2 <= 19', 'c3: x2 + 3 x3 <= 14']).", 'items': {'type': 'string'}}, 'variable_bounds': {'type': 'array', 'description': "a list of variable bounds expressed in lp file format (e.g., ['x1 >= 0', 'x2 >= 0']).", 'items': {'type': 'string'}}, 'variable_binaries': {'type': 'array', 'description': "a list of binary variables (e.g., ['x3']).", 'items': {'type': 'string'}}}, 'required': ['goal', 'objective_function', 'constraints', 'variable_bounds', 'variable_binaries']}}, 
                 {'name': 'Solve MCF with List', 'description': "useful when you need to solve a Minimum Cost Flow (MCF: find the most cost-efficient way to send a certain amount of flow through a network, subject to capacity and flow conservation constraints, while minimizing the total transportation cost.) problem given the network's structure.", 'parameters': {'type': 'object', 'properties': {'start_nodes': {'type': 'array', 'description': 'a list of starting nodes for each arc.', 'items': {'type': 'integer'}}, 'end_nodes': {'type': 'array', 'description': 'a list of ending nodes for each arc.', 'items': {'type': 'integer'}}, 'capacities': {'type': 'array', 'description': 'a list of capacities for each arc.', 'items': {'type': 'integer'}}, 'unit_costs': {'type': 'array', 'description': 'a list of unit costs for transporting flow along each arc.', 'items': {'type': 'integer'}}, 'supplies': {'type': 'array', 'description': 'a list representing the supply/demand at each node (positive for supply, negative for demand).', 'items': {'type': 'integer'}}}, 'required': ['start_nodes', 'end_nodes', 'capacities', 'unit_costs', 'supplies']}},
                 {'name': 'Solve MCF with Matrix', 'description': 'useful when you need to solve a Minimum Cost Flow (MCF: find the most cost-efficient way to send a certain amount of flow through a network, subject to capacity and flow conservation constraints, while minimizing the total transportation cost.) problem with specified capacities and costs in matrix format.', 'parameters': {'type': 'object', 'properties': {'capacity_matrix': {'type': 'array', 'description': 'a matrix representing the capacities of the edges.', 'items': {'type': 'array', 'items': {'type': 'integer'}}}, 'cost_matrix': {'type': 'array', 'description': 'a matrix representing the unit costs for the edges.', 'items': {'type': 'array', 'items': {'type': 'integer'}}}, 'supplies': {'type': 'array', 'description': 'a list representing the supply/demand at each node (positive for supply, negative for demand).', 'items': {'type': 'integer'}}}, 'required': ['capacity_matrix', 'cost_matrix', 'supplies']}}
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
    if label_tool == pred_tool:
        label_input = data['forTool']['input']['temp']
        pred_input = extract_tool_input(answer)
        pred_input = eval(f'extract_tool_mcf_input(("{pred_tool}","{pred_input}"))')
        if len(pred_input)!= len(label_input):
            return False
        else:
            if label_input==pred_input:
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
    for i in range(300000): #
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
                num_nodes = random.randint(5,8)
                num_sources = random.randint(1,3)
                num_sinks = random.randint(1,num_nodes-num_sources-1)
                minc = random.randint(10,100)
                maxc = minc * 10
                mincost = int(minc /5)
                maxcost = int(maxc /5)
                if random.random() < 0.5:
                    data = generate_min_cost_flow_list(num_nodes=num_nodes,num_sources=num_sources,num_sinks=num_sinks,min_capacity=minc,max_capacity=maxc,min_cost=mincost,max_cost=maxcost)
                else:
                    data = generate_min_cost_flow_matrix(num_nodes=num_nodes,num_sources=num_sources,num_sinks=num_sinks,min_capacity=minc,max_capacity=maxc,min_cost=mincost,max_cost=maxcost)
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
                    result['type'].append('MCF')
                    # ==================================================
                    result['question_params'].append(data['forQuestion'])
                    result['pred_input'].append(pred_input)
                    # result['pred_output'].append(answer)
                    result['judge'].append(judge_result)
                    print('已生成数据：',len(result['system']))
                    result_df = pd.DataFrame(result)
                    result_df.to_excel('/home/bigchill/mypipe/genChat5/MCF/MCF_Chat5.xlsx', index=False)
                else:
                    print('未生成数据')
                    continue
            except Exception as e:
                print('error:',e)
                # if 'error' in str(e):
                #     time.sleep(random.random())
    tasks_list = []
    for x in range(10):
        task = gevent.spawn(worker)
        tasks_list.append(task)
    gevent.joinall(tasks_list)
