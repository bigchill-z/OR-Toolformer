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
from process import extract_execute_tool,judge_init_thought
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
import pandas as pd
from tqdm import tqdm
# import swifter

def extract_tool_name(text):
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
    

# =================================================================================================================================================


pred_column = 'YOUR_PREDICTION_COLUMN'
answer_column = 'YOUR_ANSWER_COLUMN'
DATA_PATH = 'YOUR_DATA_PATH'
df = pd.read_excel(DATA_PATH)
df['output'] = df['output'].fillna('None')

#------------------------------------------
# df['name'] = 'ComoOpt'
# df['difficulty'] = '-'
# df['output'] = df['output'].apply(lambda x: eval(x)[0])
# print(df.head())
# =================================================================================================================================================

def get_answer(tool_name,tool_input, output=None):
    tool_name = tool_name.strip()
    tool_input = tool_input.strip()
    if tool_name == 'error' or tool_input == 'error':
        return 'error'
    def call_tool():
        try:
            tool = TOOL2FUNCNAME[tool_name]
            tool_output = eval(f'{tool}({tool_input})')   
            return tool_output
        except Exception as e:
            print(tool_name)
            print(tool_input)
            print(e)
            return 'error:'+str(e)
    
    if output is None:
        result = call_tool()
        for i in range(15):
            if 'HTTPSConnectionPool' not in str(result) and 'list index out of range' not in str(result):
                
                break
            else:
                time.sleep(random.random())
                result = call_tool()
            
        return result
    else:
        if 'error' in output:
            return call_tool()
        else:
            return output

def judge_fn(labels,pred):
    # # 判断输出是否正确
    print(labels,pred)
    if 'error' in str(pred):
        return 0
    else:
        if pred[0] is None:
            if labels is None or labels == 'None' or str(labels)=='nan':
                return 1
            else:
                return 0
        elif labels is None or labels == 'None' or str(labels)=='nan':
            return 0
        else:
            if abs(round(float(labels),2) - round(float(pred[0]),2))<=0.05:
                return 1
            else:
                return 0


result= {
    'user':[],
    'output':[],
    'name':[],
    'type':[],
    'difficulty':[],
    'pred': [],
    'pred_tool':[],
    'pred_tool_inputs':[],
    'pred_tool_output':[],
    'judge':[]
}
work = Queue()
for i in range(len(df)): #
    work.put_nowait(df.iloc[i])
def worker():
    while not work.empty():
        current = work.get()
        current_pred = replace_fractions_with_decimals(current[pred_column])
        # print(current_pred)
        pred_tool = extract_tool_name(current_pred)
        pred_tool_inputs = extract_tool_input(current_pred)
        
        pred_tool_output = get_answer(pred_tool, pred_tool_inputs)

        # print("pred_tool_output:",pred_tool_output)
        # pred_tool_output = get_answer(pred_tool, pred_tool_inputs, current[answer_column])
        judge = judge_fn(current[answer_column], pred_tool_output)
        # ##########################################################################
        result['user'].append(current['user'])
        ##########################################################################
        result['output'].append(current['output'])
        result['name'].append(current['name'])
        result['type'].append(current['type'])
        result['difficulty'].append(current['difficulty'])
        result['pred'].append(current_pred)
        result['pred_tool'].append(pred_tool)
        result['pred_tool_inputs'].append(pred_tool_inputs)
        result['pred_tool_output'].append(pred_tool_output)
        result['judge'].append(judge)

        print(len(result['pred_tool']))
        result_df = pd.DataFrame(result)
        result_df.to_excel(f'judge_{pred_column}_deepseekMath_5tools.xlsx', index=False)
tasks_list = []
for x in range(20):
    task = gevent.spawn(worker)
    tasks_list.append(task)
gevent.joinall(tasks_list)
# result_df 与 df 左右合并
result_df = pd.DataFrame(result)
# df = pd.concat([df, result_df], axis=1)
result_df.to_excel(f'judge_{pred_column}_deepseekMath_5tools.xlsx',index=False)

    