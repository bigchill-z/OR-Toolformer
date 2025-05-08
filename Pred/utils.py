import re


# ----------------------------------------------TSP展示路径----------------------------------------------
def showPath(path):
    result = ''
    for i in path:
        result += f' {i[0]} -> {i[1]}: {i[2]}\n'
    return result


# 示例使用
# print(showPath([[1, 5, 16], [5, 2, 45], [2, 4, 15], [4, 3, 17], [3, 1, 42]]))
# 运行结果
#  1 -> 5: 16
#  5 -> 2: 45
#  2 -> 4: 15
#  4 -> 3: 17
#  3 -> 1: 42
# ---------------------------字符串中的未定义变量转为字符串，并转换为Python数组------------------------------


def convert_undefined_to_string(input_str):
    """
    将字符串中的未定义变量和函数调用转换为字符串，并将最终的字符串转换为Python列表。

    参数:
    input_str (str): 输入的字符串，包含未定义变量和函数调用。

    返回:
    list: 处理后的Python列表。
    """

    # 使用正则表达式匹配未定义的变量和函数调用，并将其转换为字符串
    def replace_undefined(match):
        return f"'{match.group(0)}'"

    # 匹配变量名、函数调用（带括号和数组索引）
    # 允许函数名带有空格和方括号作为一部分，比如 Solve TSP with Distance Matrix()[1]
    pattern = re.compile(r'[a-zA-Z_][a-zA-Z0-9_ ]*(\(\)\[[0-9]+\]|\(\))?')

    # 替换未定义的变量和函数调用为字符串
    processed_str = re.sub(pattern, replace_undefined, input_str)

    # 将处理后的字符串转换为列表
    try:
        output_list = eval(processed_str)
    except Exception as e:
        print(f"Error evaluating the string: {e}")
        output_list = None

    return output_list


# ---------------------------------------------------从回答中提取标签用于替换-------------------------------------------------
def extract_bracket_contents(text):
    """
    从字符串中提取所有以 [ 和 ] 包裹的内容，支持嵌套的情况。
    
    参数:
    text (str): 输入字符串

    返回:
    list: 匹配的方括号内容列表
    """
    stack = []  # 用于存储方括号的开始位置
    results = []  # 存储提取的内容
    current_content = []  # 临时存储方括号内容

    # 遍历输入字符串的每个字符
    for i, char in enumerate(text):
        if char == '[':
            # 遇到左方括号，记录位置并开始新的内容
            stack.append(i)
            current_content.append('[')  # 将当前方括号添加到临时内容
        elif char == ']':
            # 遇到右方括号，取出最近的左方括号位置，并提取其中内容
            if stack:
                start = stack.pop()  # 匹配最近的左方括号
                current_content.append(']')  # 将右括号加入当前方括号内容

                # 如果栈为空，说明当前方括号是完整的一对
                if not stack:
                    # 拼接出完整的括号内容，并保存
                    results.append(''.join(current_content))
                    current_content = []  # 清空临时内容
            else:
                current_content = []  # 防止出错，清空临时内容
        elif stack:
            # 如果在方括号内部，继续累加字符到当前内容
            current_content.append(char)

    return [r for r in results if (not ('[0]' in r and '[1]' in r)) and ('[0]' in r or '[1]' in r)]


# # 测试字符串
# text = "The most efficient route for the delivery truck is [Solve TSP with Distance Matrix()[1]], which has a total distance of [Solve TSP with Distance Matrix()]"

# # 调用函数提取方括号内容
# matches = extract_bracket_contents(text)

# # 输出结果
# print(matches)

# ----------------------------------------LP 展示最优解-----------------------
def showOptimalSolution(optimal_solution):
    result = []
    for v in optimal_solution.keys():
        result.append(f'{v} = {optimal_solution[v]}')
    return '\n'.join(result)

# -------------------------------------将参数中的分数转为小数---------------------------
import re
from fractions import Fraction

def replace_fractions_with_decimals(text):
    # 定义一个函数用于将分数转换为小数
    def fraction_to_decimal(match):
        fraction_str = match.group(1)  # 获取匹配到的分数字符串（不包含括号）
        try:
            decimal_value = float(Fraction(fraction_str))  # 使用 Fraction 转换为小数
            return str(decimal_value)  # 返回小数字符串
        except ZeroDivisionError:
            return "0"  # 防止分母为零的情况
        except ValueError:
            return fraction_str # 如果不是有效分数，则返回原始字符串

    # 使用正则表达式匹配形如 "1/2" 的分数，以及"(3/4)"的括号形式
    pattern = r'\(?(\b\d+/\d+\b)\)?'  # 使用捕获组，只捕获分数部分
    result = re.sub(pattern, fraction_to_decimal, text)
    return result
# print(showOptimalSolution({'x1': '0', 'x2': '0', 'x3': '0'}))
