import os
import re
import requests
import pandas as pd


# ----------------------------------------------用于解决TSP的基础类----------------------------------------------
class TSP:
    def __init__(self, problem_type, dimension, edge_weight_type, name=None, node_coords=None, comments=None, edge_weight_section=None, edge_weight_format=None, display_data_type=None):
        """
        初始化 TSP 对象，并验证参数的有效性。
        """
        self.valid_problem_types = ['TSP', 'ATSP', 'SOP', 'HCP', 'CVRP']
        self.valid_edge_weight_types = ['EUC_2D', 'MAN_2D', 'MAX_2D', 'GEO', 'ATT', 'EXPLICIT']
        self.valid_edge_weight_formats = ['FULL_MATRIX', 'UPPER_ROW', 'LOWER_ROW', 'UPPER_DIAG_ROW', 'LOWER_DIAG_ROW']
        self.valid_display_data_types = ['COORD_DISPLAY', 'TWOD_DISPLAY']
        
        self.name = name
        self.problem_type = self.validate_parameter(problem_type, self.valid_problem_types, "problem_type")
        self.dimension = dimension
        self.edge_weight_type = self.validate_parameter(edge_weight_type, self.valid_edge_weight_types, "edge_weight_type")
        self.node_coords = node_coords
        self.comments = comments
        self.edge_weight_section = edge_weight_section
        self.edge_weight_format = self.validate_parameter(edge_weight_format, self.valid_edge_weight_formats, "edge_weight_format") if edge_weight_format else None
        self.display_data_type = self.validate_parameter(display_data_type, self.valid_display_data_types, "display_data_type") if display_data_type else None
        self.proxies = {
            "http": None,
            "https": None,
        }
    def _operate(self,data):
        url = "https://neos-server.org/neos/cgi-bin/nph-neos-solver.cgi"
        payload = self._setPayload(data)
        
        files=[]
        headers = {}
        
        response = requests.request("POST", url, headers=headers, data=payload, files=files,proxies=self.proxies).text.split('\n')[-1]
        # 正则表达式模式，用于匹配URL
        url_pattern = r'URL=(https?://[^\s">]+\.html)'
        # 使用re.findall()方法查找所有匹配的URL
        resultUrl = re.findall(url_pattern, response)[0]
        return resultUrl

    def _setPayload(self,data):
        return {'field.1': '',
        'field.2': '',
        'field.3': f'''{data}
        ''',
        'field.4': 'con',
        'field.5': 'fixed',
        'field.6': 'no',
        'field.7': '',
        'email': 'example@qq.com',
        'auto-fill': 'yes',
        'category': 'co',
        'solver': 'concorde',
        'inputMethod': 'TSP'}
    def validate_parameter(self, value, valid_options, parameter_name):
        """
        验证参数是否在候选值范围内。
        """
        if value not in valid_options:
            raise ValueError(f"Invalid {parameter_name}: {value}. Must be one of {valid_options}.")
        return value
    def sovle(self):
        data = self._generate_tsplib_format()
        url = self._operate(data)
        result2 = requests.request('GET',url,proxies=self.proxies)

        # 正则表达式模式，找到最短路程
        patternSolution = r'Optimal Solution:\s*([\d\.]+)'

        # 使用re.search找到匹配的内容
        solution = re.search(patternSolution, result2.text).group(1)

        # 正则表达式模式，找到最佳路径
        patternPlan = r'\*\*\* Cities are numbered 0\.\.n-1 and each line shows a leg from one city to the next\s+followed by the distance rounded to integers\*\*\*\n\n(.*?)</pre>'

        # 使用re.search找到匹配的内容
        plan = re.search(patternPlan, result2.text, re.DOTALL).group(1).split('\n')[1:-1]
        edges = [[int(i) for i in p.split(' ')] for p in plan]
        edges = [[i[0]+1,i[1]+1,i[2]] for i in edges]
        return {
            # 'TSPLIB':data,
            'MINIMUM DISTANCE':solution,
            'BEST ROUTE':edges
        }
    def _generate_tsplib_format(self):
        """
        生成符合 TSPLIB 格式的 TSP 文件内容。
        """
        lines = []
        
        # 添加基础信息
        if self.name:
            lines.append(f"NAME: {self.name}")
        lines.append(f"TYPE: {self.problem_type}")
        
        if self.comments:
            lines.append(f"COMMENT: {self.comments}")
            
        lines.append(f"DIMENSION: {self.dimension}")
        lines.append(f"EDGE_WEIGHT_TYPE: {self.edge_weight_type}")
        
        if self.edge_weight_format:
            lines.append(f"EDGE_WEIGHT_FORMAT: {self.edge_weight_format}")
        
        if self.display_data_type:
            lines.append(f"DISPLAY_DATA_TYPE: {self.display_data_type}")
        
        # 如果 edge_weight_section，则忽略 node_coords
        if self.edge_weight_section:
            lines.append("EDGE_WEIGHT_SECTION")
            for row in self.edge_weight_section:
                lines.append(" ".join(map(str, row)))
        elif self.node_coords:
            # 如果没有 edge_weight_section，才添加节点坐标
            lines.append("NODE_COORD_SECTION")
            for c in self.node_coords:
                coord = self.node_coords[c]
                lines.append(f"{c} {coord[0]} {coord[1]}")
        
        # 添加文件结束标识
        lines.append("EOF")
        
        return "\n".join(lines)

    def __str__(self):
        """
        返回 TSP 文件格式的内容。
        """
        return self._generate_tsplib_format()

# ----------------------------------------------使用坐标求解TSP问题----------------------------------------------
def solve_tsp_with_coordinates(num_nodes, coordinates, distance_method='EUC_2D'):
    assert num_nodes > 0
    assert type(coordinates) == dict
    assert distance_method in ['EUC_2D', 'MAX_2D', 'MAN_2D', 'GEO']
    tsp = TSP(
        problem_type="TSP",
        dimension=num_nodes,
        edge_weight_type=distance_method,
        node_coords=coordinates)
    result = tsp.sovle()
    return list(result.values()) # [MINIMUM DISTANCE,BEST ROUTE]
# 示例使用
# solve_tsp_with_coordinates(10,{1: (99, 18), 2: (13, 74), 3: (86, 44), 4: (16, 32), 5: (47, 80), 6: (69, 22), 7: (79, 2), 8: (57, 93), 9: (48, 25), 10: (91, 93)})
# 运行结果
# ['307.00',
#  [[1, 7, 26],
#   [7, 6, 22],
#   [6, 9, 21],
#   [9, 4, 33],
#   [4, 2, 42],
#   [2, 5, 35],
#   [5, 8, 16],
#   [8, 10, 34],
#   [10, 3, 49],
#   [3, 1, 29]]]

# ----------------------------------------------使用距离矩阵求解TSP问题----------------------------------------------
def solve_tsp_with_distance_matrix(num_nodes, matrix_type, matrix_data):
    assert num_nodes > 0
    assert matrix_type in ['LOWER_DIAG_ROW','FULL_MATRIX','UPPER_DIAG_ROW']
    assert type(matrix_data) == list
    tsp = TSP(
        problem_type="TSP",
        dimension=num_nodes,
        edge_weight_type="EXPLICIT",
        edge_weight_format=matrix_type,
        edge_weight_section=matrix_data)
    result = tsp.sovle()
    return list(result.values()) # [MINIMUM DISTANCE,BEST ROUTE]
# 示例使用1
# solve_tsp_with_distance_matrix(5, 'LOWER_DIAG_ROW', [[0], [4488, 0], [7197, 6059, 0], [2637, 8782, 2679, 0], [5926, 5647, 1258, 2191, 0]])
# 运行结果1
# ['16633.00', [[1, 4, 2637], [4, 5, 2191], [5, 3, 1258], [3, 2, 6059], [2, 1, 4488]]]
# 示例使用2
# solve_tsp_with_distance_matrix(5, 'UPPER_DIAG_ROW', [[0, 4745, 3128, 793, 7830], [0, 9692, 899, 4741], [0, 2028, 9747], [0, 8423], [0]])
# 运行结果2
# ['18626.00', [[1, 3, 3128], [3, 4, 2028], [4, 2, 899], [2, 5, 4741], [5, 1, 7830]]]
# 示例使用3
# solve_tsp_with_distance_matrix(5, 'FULL_MATRIX', [[0, 4030, 7202, 8888, 4693], [4030, 0, 8031, 2733, 1549], [7202, 8031, 0, 3734, 1731], [8888, 2733, 3734, 0, 1787], [4693, 1549, 1731, 1787, 0]])
# 运行结果3
# ['16921.00', [[1, 2, 4030], [2, 4, 2733], [4, 3, 3734], [3, 5, 1731], [5, 1, 4693]]]
# ----------------------------------------------使用存有坐标信息的csv文件求解TSP----------------------------------------------
def solve_tsp_from_csv(file_path,distance_method='EUC_2D'):
    assert os.path.exists(file_path)
    assert file_path.endswith('.csv')
    assert distance_method in ['EUC_2D', 'MAX_2D', 'MAN_2D', 'GEO']
    df = pd.read_csv(file_path)
    num_nodes = len(df)
    coordinates ={}
    for i in range(len(df)):
        coordinates[df['node'][i]] = (df['x'][i],df['y'][i])
    coordinates
    result = solve_tsp_with_coordinates(num_nodes, coordinates, distance_method=distance_method)
    return result # [MINIMUM DISTANCE,BEST ROUTE]
# 示例使用
# solve_tsp_from_csv('example.csv',distance_method='EUC_2D')
# 运行结果
# ['216.00', [[1, 5, 33], [5, 2, 50], [2, 4, 34], [4, 3, 25], [3, 1, 74]]]

# ----------------------------------------------使用存有坐标信息的excel文件求解TSP----------------------------------------------
def solve_tsp_from_excel(file_path,distance_method='EUC_2D'):
    assert os.path.exists(file_path)
    assert file_path.endswith('.xlsx')
    assert distance_method in ['EUC_2D', 'MAX_2D', 'MAN_2D', 'GEO']
    df = pd.read_excel(file_path)
    num_nodes = len(df)
    coordinates ={}
    for i in range(len(df)):
        coordinates[df['node'][i]] = (df['x'][i],df['y'][i])
    coordinates
    result = solve_tsp_with_coordinates(num_nodes, coordinates, distance_method=distance_method)
    return result # [MINIMUM DISTANCE,BEST ROUTE]
# 示例使用
# solve_tsp_from_excel('example.xlsx',distance_method='EUC_2D')
# 运行结果
# ['135.00', [[1, 5, 16], [5, 2, 45], [2, 4, 15], [4, 3, 17], [3, 1, 42]]]
if __name__ == "__main__":
    # result = solve_tsp_with_coordinates(18, {1: (61, 97), 2: (-48, 45), 3: (-86, 59), 4: (-26, 18), 5: (-67, -6), 6: (12, 8), 7: (63, 88), 8: (99, -21), 9: (0, -98), 10: (26, 16), 11: (-51, -21), 12: (-28, 26), 13: (-65, -9), 14: (-30, -64), 15: (-26, 40), 16: (12, -97), 17: (-75, -88), 18: (-35, -49)}, 'MAN_2D')
    result = solve_tsp_with_distance_matrix(num_nodes=3, matrix_type='FULL_MATRIX', matrix_data=[[0, 5, 3], [5, 0, 4], [3, 4, 0]])
    print(result)