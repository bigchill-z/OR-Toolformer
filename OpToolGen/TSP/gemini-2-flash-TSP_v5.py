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
You are a professional operations research teacher, skilled in designing problems related to operations research. To help students further master Traveling Salesman Problem, we now need you to set a reasonable Traveling Salesman Problem.
# OBJECTIVE #
<DATA INFORMATION>
INDUSTRY: {random.choice(['Energy', 'Health', 'Retail', 'Environment', 'Education', 'Agriculture', 'Public Utilities', 'Manufacturing', 'Software', 'Construction', 'Entertainment Legal', 'Customer Service', 'Transportation', 'Financial Services'])}
NUMBER OF LOCATIONS: {data['forQuestion']['NUMBER OF LOCATIONS']}
DATA TYPE: {data['forQuestion']['DATA TYPE']}
SPECIFIC DATA: {data['forQuestion']['SPECIFIC DATA']}
</DATA INFORMATION>
Please flexibly develop a reasonable Traveling Salesman Problem based on the above DATA INFORMATION. Prohibit the use of data other than the above-mentioned information! 
<<<Reminder requirement:
1. Create a background story based on real life.
2. On the premise of ensuring that students can understand correctly, describe the content in detail, using language that is easy to understand and avoids any ambiguity.
3. Prohibit the use of any Markdown format symbols.
4. The question should include a real background, a detailed description of the data involved, specific questions. Prohibit explicitly mentioning that this is a Traveling Salesman Problem! Prohibit explicitly mentioning the use of Traveling Salesman Problem related methods for solving!
5. Regarding the resources in the question, clear units must be provided, and "units" or "unit" should not be used in a vague manner.
6. Use <question> and </question> to package the problem description.
>>>
# RESPONSE #
A brief Traveling Salesman Problem that is easy to understand. Use <question> and </question> to package the problem. And there is no other unrelated content.
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

def generate_coordinates(num_points, x_range, y_range,type='EUC_2D',description='coordinate'):
    """
    生成带有序号、x 坐标和 y 坐标的坐标列表。

    参数:
    num_points (int): 生成的坐标数量。
    x_range (tuple): x 坐标的取值范围，例如 (min_x, max_x)。
    y_range (tuple): y 坐标的取值范围，例如 (min_y, max_y)。
    type (str): 坐标类型。
    (-10000,10000)
    EUC_2D: 二维欧几里得距离。
    MAN_2D: 二维曼哈顿距离。
    MAX_2D: 二维最大距离。
    GEO: 使用地理坐标计算的大圆距离。x:经度(-180,180),y:纬度(-90,90),建议(-30,30),(-60,60)
    Text description

    返回:
    list: 包含坐标的列表，每个坐标是一个元组 (index, x, y)。
    """
    assert type in ['EUC_2D', 'MAN_2D', 'MAX_2D', 'GEO']
    assert description in ['coordinate','text']
    type2text ={
        'EUC_2D': 'Euclidean distances in 2-D',
        'MAN_2D': 'Manhattan distances in 2-D',
        'MAX_2D': 'maximum distances in 2-D',
        'GEO': 'geographical distances',
    }
    coordinates = {}
    for i in range(1, num_points + 1):
        x = int(random.uniform(*x_range))
        y = int(random.uniform(*y_range))
        coordinates[i] = (x, y)
    SPECIFIC_DATA = ''
    
    for i in coordinates:
        if description == 'coordinate':
            SPECIFIC_DATA += f'{i} {coordinates[i][0]} {coordinates[i][1]}\n'
        else:
            SPECIFIC_DATA += f'The coordinates of location {i} are ({coordinates[i][0]},{coordinates[i][1]})\n'
    tool = 'Solve TSP with Coordinates'
    return {
        # 用于替换问题模板的内容
        'forQuestion':{
            'NUMBER OF LOCATIONS': num_points,
            'DATA TYPE': f'the location coordinates (with distances as {type2text[type]})',
            'SPECIFIC DATA': SPECIFIC_DATA,
        },
        # 用于传入API调用的参数内容
        # solve_tsp_with_coordinates(num_nodes, coordinates, distance_method='EUC_2D')
        'forTool':{
            'tool': tool,
            'input':{
                'num_nodes': num_points,
                'coordinates': coordinates,
                'distance_method': type,
                'temp':[num_points, coordinates, str(type)]
            },        
    }
}
# 示例用法
# data = generate_coordinates(5, (0, 100), (0, 100),'EUC_2D',description='text')
# data

def generate_adjacency_matrix(size, value_range=(0, 1), matrix_type='FULL_MATRIX',description='matrix'):
    """
    生成指定类型的邻接矩阵，按照给定格式。

    参数:
        size (int): 矩阵的行数（也是列数）。
        value_range (tuple): 元素的取值范围，格式为 (min_value, max_value)。
        matrix_type (str): 矩阵的类型，可选 'UPPER_DIAG_ROW'（上三角矩阵）, 'LOWER_DIAG_ROW'（下三角矩阵）, 'FULL_MATRIX'（完整矩阵）。

    返回:
        list: 生成的邻接矩阵，以嵌套列表的形式。
    """
    assert matrix_type in ['UPPER_DIAG_ROW', 'LOWER_DIAG_ROW', 'FULL_MATRIX']
    assert description in ['matrix','text']
    type2text ={
        'UPPER_DIAG_ROW': 'upper triangular matrix',
        'LOWER_DIAG_ROW': 'lower triangular matrix',
        'FULL_MATRIX': 'full matrix'
    }
    matrix = np.zeros((size, size), dtype=int)
    if description == 'text':
        matrix_type = 'LOWER_DIAG_ROW'
    
    if matrix_type == 'LOWER_DIAG_ROW':
        for i in range(1, size):
            for j in range(i):
                matrix[i, j] = np.random.randint(value_range[0], value_range[1] + 1)
    elif matrix_type == 'UPPER_DIAG_ROW':
        for i in range(size):
            for j in range(i + 1, size):
                matrix[i, j] = np.random.randint(value_range[0], value_range[1] + 1)
    elif matrix_type == 'FULL_MATRIX':
        for i in range(size):
            for j in range(size):
                if i != j:
                    matrix[i, j] = np.random.randint(value_range[0], value_range[1] + 1)
                    matrix[j, i] = matrix[i, j]        
    else:
        raise ValueError("matrix_type must be 'UPPER_DIAG_ROW', 'LOWER_DIAG_ROW', or 'FULL_MATRIX'.")
    
    # 将矩阵转换为列表形式
    # formatted_matrix = []
    # print(matrix)
    for i in range(size):
        # row = []
        if matrix_type == 'LOWER_DIAG_ROW':
            formatted_matrix = [matrix[i][:i+1].tolist() for i in range(size)]
            # path, total_distance = construct_path_lower_matrix(formatted_matrix)
        elif matrix_type == 'UPPER_DIAG_ROW':
            formatted_matrix = [matrix[i][i:].tolist() for i in range(size)]
            # path, total_distance = construct_path_upper_matrix(formatted_matrix)
        else:
            formatted_matrix = [matrix[i].tolist() for i in range(size)]
            # path, total_distance = construct_path_full_marix(formatted_matrix)
    SPECIFIC_DATA = ''
    if description == 'matrix':
        for i in range(size):
            for j in range(len(formatted_matrix[i])):
                SPECIFIC_DATA+= f"{formatted_matrix[i][j]} "
            SPECIFIC_DATA+= '\n'
        # print(SPECIFIC_DATA)
    if description == 'text':
        if matrix_type == 'LOWER_DIAG_ROW':
            for i in range(size):
                for j in range(i):
                    weight = matrix[i][j]
                    SPECIFIC_DATA+=f"The distance from location {j+1} to location {i+1} is {weight}.\n"
        # print(SPECIFIC_DATA)
    tool = 'Solve TSP with Distance Matrix'
    return {
        # 用于替换问题模板的内容
        'forQuestion':{
            'NUMBER OF LOCATIONS': size,
            'DATA TYPE': f'the distance matrix ({type2text[matrix_type]})',
            'SPECIFIC DATA': SPECIFIC_DATA,
        },
        # 用于传入API调用的参数内容
        # solve_tsp_with_distance_matrix(num_nodes, matrix_type, matrix_data)
        'forTool':{
            'tool':tool,
            'input':{
                'num_nodes': size,
                'matrix_type': matrix_type,
                'matrix_data':formatted_matrix,
                'temp':[size, matrix_type, formatted_matrix]
            }
    }
    }
    

# # 示例生成下三角矩阵
# LOWER_DIAG_ROW_matrix = generate_adjacency_matrix(5, value_range=(1, 10000), matrix_type='LOWER_DIAG_ROW',)
# print("生成的下三角矩阵:")
# LOWER_DIAG_ROW_matrix

# # # 示例生成上三角矩阵
# UPPER_DIAG_ROW_matrix = generate_adjacency_matrix(5, value_range=(1, 10000), matrix_type='UPPER_DIAG_ROW')
# # print("生成的上三角矩阵:")
# # UPPER_DIAG_ROW_matrix

# # # 示例生成完整矩阵
# FULL_MATRIX_matrix = generate_adjacency_matrix(5, value_range=(1, 10000), matrix_type='FULL_MATRIX')
# # print("生成的完整矩阵:")
# FULL_MATRIX_matrix

def get_a(question):
    pattern = r"<question>(.*?)</question>"
    q = re.search(pattern, question, re.DOTALL)
    if random.random() < 0.5:
        ques =  q.group(1).strip("\n").strip().replace("\n\n", "\n")
    else:
        ques =  q.group(1).strip("\n").strip().replace("\n\n", "")
    tool_json = [{'name': 'Solve IP', 'description': 'useful when you need to solve an Integer Programming (IP: all decision variables are required to take integer values, and both the objective function and constraints are linear.) problem defined by an objective function, constraints, variable bounds, and binary variables.', 'parameters': {'type': 'object', 'properties': {'goal': {'type': 'string', 'description': "the goal of the IP, which can be either 'Maximize' or 'Minimize'.", 'enum': ['Maximize', 'Minimize']}, 'objective_function': {'type': 'string', 'description': "the objective function expressed in lp file format (e.g., 'obj: 8 x1 + 2 x2 + 3 x3')."}, 'constraints': {'type': 'array', 'description': "a list of constraints expressed in lp file format (e.g., ['c1: 4 x1 + 3 x2 <= 28', 'c2: 3 x1 + 2 x2 <= 19', 'c3: x2 + 3 x3 <= 14']).", 'items': {'type': 'string'}}, 'variable_bounds': {'type': 'array', 'description': "a list of variable bounds expressed in lp file format (e.g., ['x1 >= 0', 'x2 >= 0']).", 'items': {'type': 'string'}}, 'variable_binaries': {'type': 'array', 'description': "a list of binary variables (e.g., ['x3']).", 'items': {'type': 'string'}}}, 'required': ['goal', 'objective_function', 'constraints', 'variable_bounds', 'variable_binaries']}}, {'name': 'Solve TSP with Distance Matrix', 'description': 'useful when you need to solve a Traveling Salesman Problem (TSP: find the shortest possible route that visits each location exactly once and returns to the origin location.) using a distance matrix representation of the nodes.', 'parameters': {'type': 'object', 'properties': {'num_nodes': {'type': 'integer', 'description': 'the number of nodes to visit.'}, 'matrix_type': {'type': 'string', 'description': 'the type of the distance matrix, which can be one of the following: "LOWER_DIAG_ROW", "FULL_MATRIX", "UPPER_DIAG_ROW".', 'enum': ['LOWER_DIAG_ROW', 'FULL_MATRIX', 'UPPER_DIAG_ROW']}, 'matrix_data': {'type': 'array', 'description': 'a list representing the distance values according to the specified matrix type.', 'items': {'type': 'array', 'items': {'type': 'number'}}}}, 'required': ['num_nodes', 'matrix_type', 'matrix_data']}},{'name': 'Solve TSP with Coordinates', 'description': 'useful when you need to solve a Traveling Salesman Problem (TSP: find the shortest possible route that visits each location exactly once and returns to the origin location.) given a set of nodes with specific coordinates.', 'parameters': {'type': 'object', 'properties': {'num_nodes': {'type': 'integer', 'description': 'the number of nodes to visit.'}, 'coordinates': {'type': 'object', 'description': 'a dictionary mapping node indices to their (x, y) coordinates, where the keys are node indices and the values are tuples representing the coordinates.', 'additionalProperties': {'type': 'array', 'items': {'type': 'number'}}}, 'distance_method': {'type': 'string', 'description': 'the method to calculate distances, which can be one of the following: "EUC_2D", "MAX_2D", "MAN_2D", "GEO" (default is "EUC_2D").', 'enum': ['EUC_2D', 'MAX_2D', 'MAN_2D', 'GEO']}}, 'required': ['num_nodes', 'coordinates']}}]
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
    
    # print(label_tool)
    # print(pred_tool)
    # print(label_input)
    # print(pred_input)
    if label_tool == pred_tool:
        label_input = data['forTool']['input']['temp']
        pred_input = extract_tool_input(answer)
        pred_input = eval(f'extract_tool_tsp_input(("{pred_tool}","{pred_input}"))')
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
            if len(result['system'])>=300:
                # 清空任务队列
                while not work.empty():
                    work.get()
                break
            try:
                if random.random() < 0.5:
                    data = generate_coordinates(num_points=random.randint(4,6),x_range=(-100,100),y_range=(-100,100),type=random.choice(['EUC_2D','EUC_2D','EUC_2D','EUC_2D','EUC_2D','EUC_2D','EUC_2D','MAN_2D','MAX_2D','GEO']),description=random.choice(['coordinate','text']))
                else:
                    data = generate_adjacency_matrix(size=random.randint(4,6),value_range=(0,10000),matrix_type=random.choice(['UPPER_DIAG_ROW', 'LOWER_DIAG_ROW', 'FULL_MATRIX']),description=random.choice(['matrix','text']))
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
                    result['type'].append('TSP')
                    # ==================================================
                    result['question_params'].append(data['forQuestion'])
                    result['pred_input'].append(pred_input)
                    # result['pred_output'].append(answer)
                    result['judge'].append(judge_result)
                    print('已生成数据：',len(result['system']))
                    result_df = pd.DataFrame(result)
                    result_df.to_excel('/home/bigchill/mypipe/genChat5/TSP/TSP_Chat5_test.xlsx', index=False)
                else:
                    continue
            except Exception as e:
                print(e)
                if 'error' in str(e):
                    time.sleep(5)
    tasks_list = []
    for x in range(10):
        task = gevent.spawn(worker)
        tasks_list.append(task)
    gevent.joinall(tasks_list)
