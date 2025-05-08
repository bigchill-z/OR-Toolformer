import os
import re
import shutil
import time
import requests
import pandas as pd
from zipfile import ZipFile


# ----------------------------用于解决混合整数线性规划的基础类----------------------------------
class MILP:
    def __init__(self, goal, objective_function: str, constraints: list, variable_bounds: list,
                 variable_binaries: list = [], variable_integers: list = []):
        assert goal in ['Maximize', 'Minimize']
        assert isinstance(objective_function, str)
        assert isinstance(constraints, list)
        assert isinstance(variable_bounds, list)
        assert isinstance(variable_binaries, list)
        assert isinstance(variable_integers, list)
        self.goal = goal
        self.objective_function = objective_function
        self.constraints = constraints
        self.variable_bounds = variable_bounds
        self.variable_binaries = variable_binaries
        self.variable_integers = variable_integers
        self.proxies = {
            "http": None,
            "https": None,
        }
        self.unzip_path = f"./{time.time()}sol"
        # 创建解压目录
        if not os.path.exists(self.unzip_path):
            os.makedirs(self.unzip_path)

    def _setPayload(self, data):
        return {
            'field.1': f'''{data}''',
            'field.2': '',
            'field.3': '',
            'field.4': '',
            'field.5': '',
            'field.6': '',
            'field.9': 'yes',
            'field.10': '',
            'priority': 'short',
            'email': 'example@qq.com',
            'auto-fill': 'yes',
            'category': 'milp',
            'solver': 'COPT',
            'inputMethod': 'LP'
        }

    def _operate(self, data):
        url = 'https://neos-server.org/neos/cgi-bin/nph-neos-solver.cgi'
        payload = self._setPayload(data)
        files = []
        headers = {}
        response = \
            requests.request("POST", url, headers=headers, data=payload, files=files, proxies=self.proxies).text.split(
                '\n')[-1]
        # 正则表达式模式，用于匹配URL
        url_pattern = r'URL=(https?://[^\s">]+\.html)'
        # 使用re.findall()方法查找所有匹配的URL
        resultUrl = re.findall(url_pattern, response)[0]
        return resultUrl

    def _generate_milp_format(self, ):
        "生成符号MILP格式的文件内容"
        lines = []
        # 添加目标函数类型
        lines.append(self.goal)
        # 添加目标函数
        lines.append(self.objective_function)
        # 添加约束条件
        lines.append('Subject To')
        for c in self.constraints:
            lines.append(c)
        # 添加变量边界
        lines.append('Bounds')
        for b in self.variable_bounds:
            lines.append(b)
        # 添加01变量声明
        if self.variable_binaries:
            lines.append("Binaries")
            for v in self.variable_binaries:
                lines.append(v)
        # 添加整数变量声明
        if self.variable_integers:
            lines.append('Integers')
            for v in self.variable_integers:
                lines.append(v)

        # 添加文件结束标识
        lines.append('End')
        return '\n'.join(lines)

    def sovle(self):
        try:
            data = self._generate_milp_format()
            url = self._operate(data)
            result2 = requests.request('GET', url, proxies=self.proxies)
            # # 正则表达式模式，找到最佳目标值
            # patternSolution = r"Objective:\s*(\S+)"
            # # 使用re.search找到匹配的内容
            # solution = re.search(patternSolution, result2.text).group(1)
            # solution = eval(solution)
            # 正则表达式模式，找到zip下载地址
            patternZipPath = r'Additional Output: <br/><a href="(https://.+?)"'
            zipPath = re.search(patternZipPath, result2.text).group(1)

            response = requests.get(zipPath, proxies=self.proxies)
            zipFilePath = os.path.join(self.unzip_path, 'sol.zip')
            with open(zipFilePath, 'wb') as f:
                f.write(response.content)
            print(f"下载成功: {zipFilePath}")
            # 解压文件
            with ZipFile(zipFilePath, 'r') as zip_ref:
                zip_ref.extractall(self.unzip_path)

            print(f"解压完成: {zipFilePath} 到 {self.unzip_path}")
            # 如果无解就没有sol文件 直接结束
            if not os.path.exists(self.unzip_path + '/soln.sol'):
                shutil.rmtree(self.unzip_path)
                print(f"删除解压文件: {self.unzip_path}")
                return None, None
            # 解析 soln.sol文件
            with open(self.unzip_path + '/soln.sol', 'r') as f:
                sol = f.readlines()
            objective_value = sol[0].replace('\n', '').split(' ')[-1]
            variable = {}
            for v in sol[1:]:
                value = v.replace('\n', '').split(' ')
                variable[value[0]] = value[-1]
            shutil.rmtree(self.unzip_path)
            print(f"删除解压文件: {self.unzip_path}")
            return objective_value, variable
        finally:
            if os.path.exists(self.unzip_path):
                shutil.rmtree(self.unzip_path)
                print(f"删除解压文件: {self.unzip_path}")
            # return 'error:'+str(e)

    def __str__(self):
        """
        返回MILP文件格式内容
        """
        return self._generate_milp_format()


# ----------------------------------------求解MILP问题
def solve_milp(goal, objective_function, constraints, variable_bounds, variable_binaries, variable_integers):
    # 每部分参数都要符合LP文件格式
    assert goal in ['Maximize', 'Minimize']
    assert isinstance(objective_function, str)
    assert isinstance(constraints, list)
    assert isinstance(variable_bounds, list)
    assert isinstance(variable_binaries, list)
    assert isinstance(variable_integers, list)
    milp = MILP(goal, objective_function, constraints, variable_bounds, variable_binaries, variable_integers)
    return list(milp.sovle())


# ----------------------------------------求解MILP问题
def solve_milp(goal, objective_function, constraints, variable_bounds, variable_binaries, variable_integers):
    # 每部分参数都要符合LP文件格式
    assert goal in ['Maximize', 'Minimize']
    assert isinstance(objective_function, str)
    assert isinstance(constraints, list)
    assert isinstance(variable_bounds, list)
    assert isinstance(variable_binaries, list)
    assert isinstance(variable_integers, list)
    milp = MILP(goal, objective_function, constraints, variable_bounds, variable_binaries, variable_integers)
    return list(milp.sovle())


# 示例使用
# print(solve_milp('Maximize', 'obj: 31 x1 + 44 x2', ['c1: 66 x1 + 25 x2 <= 1292', 'c2: 61 x1 + 56 x2 <= 1254', 'c3: 0.7 x2 - 0.3 x1 <= 0'], ['x1 >= 0', 'x2 >= 0'], [], ['x1', 'x2']))
# 运行结果
# ['45.5', {'x1': '7', 'x2': '4', 'x3': '4.5', 'x4': '0'}]
if __name__ == '__main__':
#     candi = """'Minimize', 'obj: 5 x1 + 4 x2 + 3 x3', ['c1: 2 x1 + x2 - x3 <= 1000', 'c2: 3 x1 - x2 + x3 >= 500', 'c3: - x1 + 4 x2 - 2 x3 = 200'], ['x1 >= 0', 'x2 >= 0', 'x3 >= 0'], [], ['x1', 'x2', 'x3']
# 'Minimize', 'obj: 5 x1 + 6 x2 + 7 x3 + 8 x4', ['c1: x1 - x2 <= 5', 'c2: x3 - x4 >= 3', 'c3: x1 + x2 + x3 + x4 <= 10', 'c4: x1 - 2 <= 0', 'c5: x2 - 4 <= 0', 'c6: 10 <= x1 <= 5', 'c7: 5 <= x2 <= 3', 'c8: x3 - 2.5 <= 0', 'c9: x4 - 1.5 <= 0'], ['x1 >= 0', 'x2 >= 0', 'x3 >= 0', 'x4 >= 0'], [], ['x1', 'x2', 'x3', 'x4']"""
#     result = []
#     for i in candi.split("\n"):
#         try:
#             result.append(eval(f'solve_milp({i})'))
#         except Exception as e:
#             result.append(f'error:{e}')
#     for r in result:
#         print(r)
    result = solve_milp(goal='Minimize', objective_function='obj: 20 x12 + 39 x13 + 4 x14 + 45 x15 + x21 + 8 x23 + 23 x24 + 44 x25 + 5 x31 + 46 x32 + 36 x34 + 15 x35 + 3 x41 + 5 x42 + 10 x43 + 27 x45 + 37 x51 + 15 x52 + 7 x53 + 45 x54', constraints=['c1: x12 + x13 + x14 + x15 - x21 - x31 - x41 - x51 <= 93', 'c2: x21 + x23 + x24 + x25 - x12 - x32 - x42 - x52 <= -219', 'c3: x31 + x32 + x34 + x35 - x13 - x23 - x43 - x53 <= -57', 'c4: x41 + x42 + x43 + x45 - x14 - x24 - x34 - x54 <= 317', 'c5: x51 + x52 + x53 + x54 - x15 - x25 - x35 - x45 <= 149', 'c6: -x12 - x13 - x14 - x15 + x21 + x31 + x41 + x51 <= -93', 'c7: -x21 - x23 - x24 - x25 + x12 + x32 + x42 + x52 <= 219', 'c8: -x31 - x32 - x34 - x35 + x13 + x23 + x43 + x53 <= 57', 'c9: -x41 - x42 - x43 - x45 + x14 + x24 + x34 + x54 <= -317', 'c10: -x51 - x52 - x53 - x54 + x15 + x25 + x35 + x45 <= -149'], variable_bounds=['x12 >= 0', 'x13 >= 0', 'x14 >= 0', 'x15 >= 0', 'x21 >= 0', 'x23 >= 0', 'x24 >= 0', 'x25 >= 0', 'x31 >= 0', 'x32 >= 0', 'x34 >= 0', 'x35 >= 0', 'x41 >= 0', 'x42 >= 0', 'x43 >= 0', 'x45 >= 0', 'x51 >= 0', 'x52 >= 0', 'x53 >= 0', 'x54 >= 0'], variable_binaries=[], variable_integers=['x12', 'x13', 'x14', 'x15', 'x21', 'x23', 'x24', 'x25', 'x31', 'x32', 'x34', 'x35', 'x41', 'x42', 'x43', 'x45', 'x51', 'x52', 'x53', 'x54'])
    print(result)