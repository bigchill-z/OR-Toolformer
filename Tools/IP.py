import os
import re
import shutil
import time
import requests
import pandas as pd
from zipfile import ZipFile


# ----------------------------用于解决混合整数线性规划的基础类----------------------------------
class IP:
    def __init__(self, goal, objective_function: str, constraints: list, variable_bounds: list,
                 variable_binaries: list = []):
        assert goal in ['Maximize', 'Minimize']
        assert isinstance(objective_function, str)
        assert isinstance(constraints, list)
        assert isinstance(variable_bounds, list)
        assert isinstance(variable_binaries, list)
        self.goal = goal
        self.objective_function = objective_function
        self.constraints = constraints
        self.variable_bounds = variable_bounds
        self.variable_binaries = variable_binaries
        # 查找变量
        self.variable_integers = [v for v in re.findall(r'\b(x\d+)', objective_function) if v not in self.variable_binaries]
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
            # print(self.objective_function)
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


# ----------------------------------------求解IP问题
def solve_ip(goal, objective_function, constraints, variable_bounds, variable_binaries):
    # 每部分参数都要符合LP文件格式
    assert goal in ['Maximize', 'Minimize']
    assert isinstance(objective_function, str)
    assert isinstance(constraints, list)
    assert isinstance(variable_bounds, list)
    assert isinstance(variable_binaries, list)
    milp = IP(goal, objective_function, constraints, variable_bounds, variable_binaries)
    # print(milp)
    return list(milp.sovle())

if __name__ =='__main__':
    # result = solve_ip('Maximize', 'obj: 6 x1 + 36 x2 + 27 x3', ['c1: 28 x1 + 29 x2 + 18 x3 <= 190', 'c2: 26 x1 + 12 x2 + 12 x3 <= 361', 'c3: 23 x1 + 17 x2 + 13 x3 <= 627', 'c4: 20 x1 + 14 x2 + 36 x3 <= 361'], ['x1 >= 0', 'x2 >= 0', 'x3 >= 0'], [])
    # print(result)
    # candi = """'Maximize', 'obj: 12 x1 + 5 x2', ['c1: 10 x1 + 7 x2 <= 1000', 'c2: 5 x2 - x1 <= 0', 'c3: x1 >= 10'], ['x1 >= 0', 'x2 >= 0'], []"""
    # result = []
    # for i in candi.split("\n"):
    #     try:
    #         result.append(eval(f'solve_ip({i})'))
    #     except Exception as e:
    #         result.append(f'error:{e}')
    # for r in result:
    #     print(r)
    result = solve_ip(goal='Minimize', objective_function='obj: 10 x1 + 20 x2 + 30 x3 + 40 x4', constraints=['c1: x1 + x2 <= 500', 'c2: x3 + x4 <= 600', 'c3: x1 - x3 >= 100', 'c4: x2 - x4 >= 200'], variable_bounds=['x1 >= 0', 'x2 >= 0', 'x3 >= 0', 'x4 >= 0'], variable_binaries=[])
    print(result)