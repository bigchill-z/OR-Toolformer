import re
import random

def extract_tool_name(text):
    '''
    抽取工具名
    '''
    result = 'error'
    pattern = r"<use_tool>(.*?)\("
    # 查找匹配的内容
    try:
        name = re.search(pattern, text, re.DOTALL)
        if name is not None :
            result = name.group(1).strip()
            return result
        else:
            return result
    except:
        return result
def extract_tool_input(text):
    '''
    抽取工具输入
    '''
    result = 'error'
    pattern = r"<use_tool>.*\((.*?)\)</use_tool>"# <\/use_tool>
    # 查找匹配的内容
    try:
        input = re.search(pattern, text, re.DOTALL)
        if input is not None:
            result = input.group(1).strip()
            return result
        else:
            return result
    except:
            return result
def extract_tool_lp_input(goal,objective_function,constraints,variable_bounds):
    '''抽取gemini生成的答案中lp参数'''
    try:
        return [goal,objective_function,constraints,variable_bounds]
    except:
        return 'error'
def extract_tool_ip_input(goal,objective_function,constraints,variable_bounds,variable_binaries):
    '''抽取gemini生成的答案中ip参数'''
    try:
        return [goal,objective_function,constraints,variable_bounds,variable_binaries]
    except:
        return 'error'
def extract_tool_milp_input(goal,objective_function,constraints,variable_bounds,variable_binaries,variable_integers):
    '''抽取gemini生成的答案中milp参数'''
    try:
        return [goal,objective_function,constraints,variable_bounds,variable_binaries,variable_integers]
    except:
        return 'error'
def extract_tool_tsp_input(name_input):
    # 断言 name_input是一个元组
    assert type(name_input) == tuple
    tool_name = name_input[0]
    tool_input = name_input[1:]
    temp = [*tool_input][0]
    def extract_tool_tsp_matrix(num_nodes, matrix_type, matrix_data):
        return [num_nodes, matrix_type, matrix_data]
    def extract_tool_tsp_coordinates(num_nodes, coordinates,distance_method='EUC_2D'):
        return [num_nodes, coordinates, distance_method]
    
    if tool_name == 'Solve TSP with Distance Matrix':
        return eval(f'extract_tool_tsp_matrix({temp})')
    elif tool_name == 'Solve TSP with Coordinates':
        return eval(f'extract_tool_tsp_coordinates({temp})')
    else:
        return 'error'
    
# extract_tool_tsp_input(('Solve TSP with Coordinates',"num_nodes=5, coordinates={'1': [29, 94], '2': [42, 70], '3': [32, 72], '4': [89, 67], '5': [90, 81]}, distance_method='EUC_2D'"))
def extract_tool_mf_input(name_input):
    # 断言 name_input是一个元组
    assert type(name_input) == tuple
    tool_name = name_input[0]
    tool_input = name_input[1:]
    temp = [*tool_input][0]
    # print(temp)
    def extract_tool_mf_matrix(matrix):
        return [matrix]
    def extract_tool_mf_list(start_nodes, end_nodes, capacities):
        return [start_nodes, end_nodes, capacities]
    
    if tool_name == 'Solve MF with Matrix':
        return eval(f'extract_tool_mf_matrix({temp})')
    elif tool_name == 'Solve MF with List':
        return eval(f'extract_tool_mf_list({temp})')
    else:
        return 'error'
    
# extract_tool_mf_input(('Solve MF with List','{"start_nodes": [0, 2, 2], "end_nodes": [1, 1, 3], "capacities": [415, 162, 247]}'))
    
# extract_tool_mf_input(('Solve MF with Matrix',"[[0.0, 8.0, 10.0, 1.0, 8.0, 2.0],[0.0, 0.0, 10.0, 2.0, 0.0, 6.0],[0.0, 0.0, 0.0, 7.0, 8.0, 6.0],[0.0, 10.0, 6.0, 0.0, 2.0, 10.0],[0.0, 0.0, 4.0, 3.0, 0.0, 3.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]"))
def extract_tool_ap_input(name_input):
    # 断言 name_input是一个元组
    assert type(name_input) == tuple
    tool_name = name_input[0]
    tool_input = name_input[1:]
    temp = [*tool_input][0]
    # print(temp)
    def extract_tool_ap_matrix(goal, matrix_data, agents=None, tasks=None):
        return [goal, matrix_data, agents, tasks]
    def extract_tool_ap_list(goal, agents, tasks, weights):
        return [goal, agents, tasks, weights]
    
    if tool_name == 'Solve AP with Matrix':
        return eval(f'extract_tool_ap_matrix({temp})')
    elif tool_name == 'Solve AP with List':
        return eval(f'extract_tool_ap_list({temp})')
    else:
        return 'error'
    
# extract_tool_ap_input(('Solve AP with Matrix',"goal='Minimize', matrix_data=[[100, 106, 73, 0], [86, 95, 66, 0], [119, 135, 99, 0], [86, 65, 82, 0]], agents=['Truck 1', 'Truck 2', 'Truck 3', 'Truck 4'], tasks=['A', 'B', 'C', 'Dummy']"))
def extract_tool_mcf_input(name_input):
    # 断言 name_input是一个元组
    assert type(name_input) == tuple
    tool_name = name_input[0]
    tool_input = name_input[1:]
    temp = [*tool_input][0]
    print(temp)
    def extract_tool_mcf_matrix(capacity_matrix, cost_matrix, supplies):
        return [capacity_matrix, cost_matrix, supplies]
    def extract_tool_mcf_list(start_nodes, end_nodes, capacities, unit_costs,supplies):
        return [start_nodes, end_nodes, capacities, unit_costs,supplies]
    
    if tool_name == 'Solve MCF with Matrix':
        return eval(f'extract_tool_mcf_matrix({temp})')
    elif tool_name == 'Solve MCF with List':
        # temp = eval(temp)
        return eval(f'extract_tool_mcf_list({temp})')
    else:
        return 'error'
    
# extract_tool_mcf_input(('Solve MCF with List','start_nodes=[0, 0, 1, 1, 2, 3], end_nodes=[2, 3, 3, 5, 5, 2], capacities=[27, 77, 97, 42, 82, 51], unit_costs=[1, 6, 6, 19, 19, 5], supplies=[10, 131, 0, 0, -36, -105]'))

# 随机选择工具名,但是保证必须的工具,至少3个工具,最多5个工具
def select_tools(necessary_tools, candidate_tools, x=3, y=5):
    # 保证返回的工具名列表包含必要的工具
    selected_tools = necessary_tools.copy()
    x = 0 if len(necessary_tools) > x else x - len(necessary_tools)
    y = y - len(necessary_tools)
    num = random.randint(x, y)
    t = 0
    # 在候选工具中随机选择，直到满足数量要求
    while t < num:
        tool = random.choice(candidate_tools)
        if tool not in selected_tools:
            selected_tools.append(tool)
            t += 1

    # 去重后打乱顺序
    selected_tools = list(set(selected_tools))
    random.shuffle(selected_tools)

    return selected_tools
# print(len(select_tools(['1'],['1','2','3','4','5','6','7'],4,4)))
def exclude_necessary_tools(necessary_tools, candidate_tools,x=5, y=5):
    # 确定需要排除的工具
    if "Solve LP" in necessary_tools:
        excluded_tools = ["Solve LP", "Solve MILP"]
    elif "Solve IP" in necessary_tools:
        excluded_tools = ["Solve IP", "Solve MILP"]
    elif "Solve MILP" in necessary_tools:
        excluded_tools = ["Solve MILP"]
    elif "Solve TSP with Coordinates" in necessary_tools or "Solve TSP with Distance Matrix" in necessary_tools:
        excluded_tools = ["Solve TSP with Coordinates", "Solve TSP with Distance Matrix"]
    elif "Solve MF with List" in necessary_tools or "Solve MF with Matrix" in necessary_tools:
        excluded_tools = ["Solve MF with List", "Solve MF with Matrix"]
    else:
        excluded_tools = []

    # 从候选工具中排除必要的工具
    selected_tools = [tool for tool in candidate_tools if tool not in excluded_tools]
    
    x = min(x, len(selected_tools))
    y = min(y, len(selected_tools))
    return random.sample(selected_tools, random.randint(x, y))
def clear_assistant(text):
    result = []
    text_list = text.split("\n")
    for t in text_list:
        # for tool in ['`Solve LP`', '`Solve IP`', '`Solve MILP`', '*Solve LP*', '*Solve IP*', '*Solve MILP*']:
        #     temp = t.replace(tool,f'"{tool}"')
        temp = t.replace('*','').replace('`','').replace('-','').strip()
        result.append(temp)

    return '\n'.join(result)
if __name__ == '__main__':
    # print(len(select_tools(['1'],['1','2','3','4','5','6','7'],4,4)))
    print(exclude_necessary_tools(['Solve MILP'], ['Solve LP', 'Solve IP', 'Solve MILP', 'Solve TSP with Coordinates', 'Solve TSP with Distance Matrix', 'Solve MF with List', 'Solve MF with Matrix']))


