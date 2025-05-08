# ----------------------------------------------工具描述----------------------------------------------
    # 'Solve TSP with Coordinates': 'useful when you need to solve a Traveling Salesman Problem (TSP) given a set of nodes with specific coordinates. The input to this function should include the number of nodes as an integer, a dictionary representing the coordinates of each node, and an optional string representing the distance calculation method ("EUC_2D", "MAX_2D", "MAN_2D", or "GEO"). The function outputs a list containing two elements: the minimum distance as a string, and the best route as a list of lists (with each sublist representing an edge between two nodes and its distance).',
    # 'Solve TSP with Distance Matrix': 'useful when you need to solve a Traveling Salesman Problem (TSP) using a pre-defined distance matrix. The input to this function should include the number of nodes as an integer, a string representing the type of the matrix ("LOWER_DIAG_ROW", "FULL_MATRIX", or "UPPER_DIAG_ROW"), and a list of lists representing the distance matrix data. The function outputs a list containing two elements: the minimum distance as a string, and the best route as a list of lists (with each sublist representing an edge between two nodes and its distance).',
    # 'Draw Directed Graph': 'useful when you need to visualize a directed graph with optional node coordinates. The input to this function should be a list of edges, where each edge is represented as a sublist with three elements: the start node, the end node, and the weight of the edge. Optionally, you can provide a dictionary of node coordinates, where the keys are node identifiers and the values are tuples representing the (x, y) coordinates of the nodes. The function outputs the file path of the saved image, which is a PNG file showing the directed graph with labeled nodes and edges.',
    # 'Solve LP': 'useful when you need to solve a linear programming (LP) problem by specifying the goal (maximize or minimize), the objective function, constraints, and variable bounds. The input to this function should include the goal type, a string representing the objective function, a list of constraints, and a list of variable bounds. The output is a list containing the optimal value of the objective function and a dictionary of variable values at that optimal solution.',    
    # 'Solve MILP': 'useful when you need to solve a mixed integer linear programming (MILP) problem by specifying the goal (maximize or minimize), the objective function, constraints, variable bounds, binary variables, and integer variables. The input to this function should include the goal type, a string representing the objective function, a list of constraints, a list of variable bounds, a list of binary variables, and a list of integer variables. The output is a list containing the optimal value of the objective function and a dictionary of variable values at that optimal solution.',
    # 'Solve IP': 'useful when you need to solve an integer programming (IP) problem defined by an objective function, constraints, variable bounds, and binary variables. The input includes the goal (maximize or minimize), the objective function as a string, a list of constraints, bounds for the variables, and a list of binary variables. The output consists of the optimal value and a dictionary of variable values.',
    # 'Solve NP': 'useful when you need to solve a nonlinear programming (NLP) problem defined by an objective function and constraints. The input includes the goal (maximize or minimize), the objective function as a string, a list of constraints, and variable bounds. The output consists of the optimal value and a dictionary of variable values.',
    # 'Solve MF with List': 'useful when you need to solve a maximum flow (MF) problem in a network defined by start and end nodes along with their respective capacities. The input to this function should include arrays for start nodes, end nodes, and capacities. The output is a list containing the maximum flow value and a list of flows for each edge, detailing the start node, end node, flow, and capacity.',
    # 'Solve MF with Matrix': 'useful when you need to solve a maximum flow (MF) problem represented by a capacity matrix, where non-zero values indicate the presence of edges between nodes. The input to this function should be a 2D matrix where each entry represents the capacity of the edge between nodes. The output is a list containing the maximum flow value and a list of flows for each edge, detailing the start node, end node, flow, and capacity.',
    # 'Solve AP with Matrix': 'useful when you need to solve an assignment problem (AP) where agents are assigned to tasks based on a cost matrix, and you want to minimize or maximize the total cost or benefit. The input to this function should include the goal (minimize or maximize), a matrix of costs or benefits, and optionally lists of agent and task names. The output is a list containing the optimal objective value and the optimal assignment of agents to tasks with the associated costs or benefits.',
    # 'Solve AP with List': 'useful when you need to solve an assignment problem (AP) using lists of agents, tasks, and corresponding weights. The input should include the goal (either "Minimize" or "Maximize"), a list of agents, a list of tasks, and a list of weights representing the cost or benefit of each assignment. The output is a list containing the optimal value and a list of optimal assignments, detailing which agent is assigned to which task along with the associated weight.',

# 生成chat数据时工具集的数量
x = 5
y = 5

TOOLS_DESCRIPTION = {
    'Solve TSP with Coordinates': '''useful when you need to solve a Traveling Salesman Problem (TSP: find the shortest possible route that visits each location exactly once and returns to the origin location.) given a set of nodes with specific coordinates. The input to this tool should include the following parameters: 
- num_nodes (int): the number of nodes to visit.
- coordinates (dict): a dictionary mapping node indices to their (x, y) coordinates, where the keys are node indices and the values are tuples representing the coordinates.
- distance_method (str, optional): the method to calculate distances, which can be one of the following: 
  - "EUC_2D" for Euclidean distance
  - "MAX_2D" for maximum distance
  - "MAN_2D" for Manhattan distance
  - "GEO" for geographical distance (default is "EUC_2D").
The output will be a list containing the minimum distance and the best route taken as a list of edges.
Example usage: Solve TSP with Coordinates(10, {1: (99, 18), 2: (13, 74), 3: (86, 44), 4: (16, 32), 5: (47, 80), 6: (69, 22), 7: (79, 2), 8: (57, 93), 9: (48, 25), 10: (91, 93)}) will yield the result ['307.00', [[1, 7, 26], [7, 6, 22], [6, 9, 21], [9, 4, 33], [4, 2, 42], [2, 5, 35], [5, 8, 16], [8, 10, 34], [10, 3, 49], [3, 1, 29]]].''',
    
    'Solve TSP with Distance Matrix': '''useful when you need to solve a Traveling Salesman Problem (TSP: find the shortest possible route that visits each location exactly once and returns to the origin location.) using a distance matrix representation of the nodes. The input to this tool should include the following parameters: 
- num_nodes (int): the number of nodes to visit.
- matrix_type (str): the type of the distance matrix, which can be one of the following:
  - "LOWER_DIAG_ROW" for lower triangular matrix format,
  - "FULL_MATRIX" for full matrix format,
  - "UPPER_DIAG_ROW" for upper triangular matrix format.
- matrix_data (list): a list representing the distance values according to the specified matrix type.
The output will be a list containing the minimum distance and the best route taken as a list of edges.
Example usage1: Solve TSP with Distance Matrix(5, 'LOWER_DIAG_ROW', [[0], [4488, 0], [7197, 6059, 0], [2637, 8782, 2679, 0], [5926, 5647, 1258, 2191, 0]]) will yield the result ['16633.00', [[1, 4, 2637], [4, 5, 2191], [5, 3, 1258], [3, 2, 6059], [2, 1, 4488]]].
Example usage2: Solve TSP with Distance Matrix(5, 'UPPER_DIAG_ROW', [[0, 4745, 3128, 793, 7830], [0, 9692, 899, 4741], [0, 2028, 9747], [0, 8423], [0]]) will yield the result ['18626.00', [[1, 3, 3128], [3, 4, 2028], [4, 2, 899], [2, 5, 4741], [5, 1, 7830]]].
Example usage3: Solve TSP with Distance Matrix(5, 'FULL_MATRIX', [[0, 4030, 7202, 8888, 4693], [4030, 0, 8031, 2733, 1549], [7202, 8031, 0, 3734, 1731], [8888, 2733, 3734, 0, 1787], [4693, 1549, 1731, 1787, 0]]) will yield the result ['16921.00', [[1, 2, 4030], [2, 4, 2733], [4, 3, 3734], [3, 5, 1731], [5, 1, 4693]]].''',

    'Solve LP': '''useful when you need to solve a Linear Programming (LP: find the best outcome in a model with linear relationships, subject to a set of constraints.) problem defined by an objective function, constraints, and variable bounds. The input to this tool should include the following parameters:
- goal (str): the goal of the LP, which can be either 'Maximize' or 'Minimize'.
- objective_function (str): the objective function expressed in lp file format (e.g., 'obj: 8 x + 15 y').
- constraints (list): a list of constraints expressed in lp file format (e.g., ['c1: 10 x + 15 y <= 3000', 'c2: x + y <= 250']).
- variable_bounds (list): a list of variable bounds expressed in lp file format (e.g., ['x >= 0', 'y >= 0']).
The output will be a list containing the maximum or minimum value and a dictionary of variable values.
Example usage: Solve LP('Maximize', 'obj: 8 x + 15 y', ['c1: 10 x + 15 y <= 3000', 'c2: x + y <= 250'], ['x >= 0', 'y >= 0']) will yield the result ['3000', {'x': '0', 'y': '200'}].''',

    'Solve MILP': '''useful when you need to solve a Mixed Integer Linear Programming (MILP: the objective function and constraints are linear, but some of the decision variables are restricted to integer values, while others can be continuous.) problem defined by an objective function, constraints, variable bounds, binary variables, and integer variables. The input to this tool should include the following parameters:
- goal (str): the goal of the MILP, which can be either 'Maximize' or 'Minimize'.
- objective_function (str): the objective function expressed in lp file format (e.g., 'obj: 3 x1 + 5 x2 + x3 + x4').
- constraints (list): a list of constraints expressed in lp file format (e.g., ['c1: 2 x1 + x2 + x4 <= 18.5', 'c2: x1 + 2 x2 <= 15', 'c3: x2 + x3 <= 8.5']).
- variable_bounds (list): a list of variable bounds expressed in lp file format (e.g., ['x1 >= 0', 'x2 >= 0', 'x3 >= 0']).
- variable_binaries (list): a list of binary variables (e.g., ['x4']).
- variable_integers (list): a list of integer variables (e.g., ['x1', 'x2']).
The output will be a list containing the maximum or minimum value and a dictionary of variable values.
Example usage: Solve MILP('Maximize', 'obj: 3 x1 + 5 x2 + x3 + x4', ['c1: 2 x1 + x2 + x4 <= 18.5', 'c2: x1 + 2 x2 <= 15', 'c3: x2 + x3 <= 8.5'], ['x1 >= 0', 'x2 >= 0', 'x3 >= 0'], ['x4'], ['x1', 'x2']) will yield the result ['45.5', {'x1': '7', 'x2': '4', 'x3': '4.5', 'x4': '0'}].''',
    
    'Solve IP': '''useful when you need to solve an Integer Programming (IP: all decision variables are required to take integer values, and both the objective function and constraints are linear.) problem defined by an objective function, constraints, variable bounds, and binary variables. The input to this tool should include the following parameters:
- goal (str): the goal of the IP, which can be either 'Maximize' or 'Minimize'.
- objective_function (str): the objective function expressed in lp file format (e.g., 'obj: 8 x1 + 2 x2 + 3 x3').
- constraints (list): a list of constraints expressed in lp file format (e.g., ['c1: 4 x1 + 3 x2 <= 28', 'c2: 3 x1 + 2 x2 <= 19', 'c3: x2 + 3 x3 <= 14']).
- variable_bounds (list): a list of variable bounds expressed in lp file format (e.g., ['x1 >= 0', 'x2 >= 0']).
- variable_binaries (list): a list of binary variables (e.g., ['x3']).
The output will be a list containing the maximum or minimum value and a dictionary of variable values.
Example usage: Solve IP('Maximize', 'obj: 8 x1 + 2 x2 + 3 x3', ['c1: 4 x1 + 3 x2 <= 28', 'c2: 3 x1 + 2 x2 <= 19', 'c3: x2 + 3 x3 <= 14'], ['x1 >= 0', 'x2 >= 0'], ['x3']) will yield the result ['51', {'x1': '6', 'x2': '0', 'x3': '1'}].''',

    'Solve MF with List': '''useful when you need to solve a Maximum Flow (MF: find the greatest possible flow of resources from a source node to a sink node in a network, while respecting the capacities of the edges.) problem using a list representation of edges. The input to this tool should include the following parameters:
- start_nodes (list): a one-dimensional array representing the starting nodes of each edge.
- end_nodes (list): a one-dimensional array representing the ending nodes of each edge.
- capacities (list): a one-dimensional array representing the capacity limits for each edge.
The output will be a list containing the maximum flow value and a list of flow distributions for each edge in the format [start_node, end_node, flow, capacity].
Example usage: Solve MF with List([0, 0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 3, 2, 4, 3, 4, 2, 4], [20, 30, 10, 40, 30, 10, 20, 5, 20]) will yield the result [60, [[0, 1, 20, 20], [0, 2, 30, 30], [0, 3, 10, 10], [1, 2, 0, 40], [1, 4, 20, 30], [2, 3, 10, 10], [2, 4, 20, 20], [3, 2, 0, 5], [3, 4, 20, 20]]].''',

    'Solve MF with Matrix': '''useful when you need to solve a Maximum Flow (MF: find the greatest possible flow of resources from a source node to a sink node in a network, while respecting the capacities of the edges.) problem using a matrix representation of the capacities. The input to this tool should include the following parameter:
- matrix (list of lists): a 2D array where non-zero values represent the capacity of the edges between nodes. The element at position [i][j] indicates the capacity from node i to node j.
The output will be a list containing the maximum flow value and a list of flow distributions for each edge in the format [start_node, end_node, flow, capacity].
Example usage: Solve MF with Matrix([[0, 8, 10, 1, 8, 2], [0, 0, 10, 2, 0, 6], [0, 0, 0, 7, 8, 6], [0, 10, 6, 0, 2, 10], [0, 0, 4, 3, 0, 3], [0, 0, 0, 0, 0, 0]]) will yield the result [60, [[0, 1, 20, 20], [0, 2, 30, 30], [0, 3, 10, 10], [1, 2, 0, 40], [1, 4, 20, 30], [2, 3, 10, 10], [2, 4, 20, 20], [3, 2, 0, 5], [3, 4, 20, 20]]].''',
}
TOOLS_DESCRIPTION_PLUS = {
    'Solve AP with Matrix': '''useful when you need to solve an Assignment Problem (AP: the goal is to assign a set of tasks to a set of agents in such a way that minimizes the total cost or maximizes the total profit, subject to the limitation that each task is assigned to exactly one agent and each agent is assigned exactly one task.) using a matrix representation of costs or benefits. The input to this tool should include the following parameters:
- goal (str): the goal of the assignment problem, which can be either 'Maximize' or 'Minimize'.
- matrix_data (list of lists): a 2D array where each element represents the cost (for minimization) or benefit (for maximization) of assigning agent i to task j.
- agents (list, optional): a list of agent identifiers (e.g., ['w1', 'w2', 'w3', 'w4', 'w5']).
- tasks (list, optional): a list of task identifiers (e.g., ['t1', 't2', 't3', 't4']).
The output will be a list containing the optimal value and a list of optimal assignments in the format [agent, task, cost/benefit].
Example usage: Solve AP with Matrix("Minimize", [[90.5, 80, 75, 70], [35, 85, 55.3, 65], [125, 95, 90, 95], [45, 110.2, 95, 115], [50, 100, 90, 100]], ['w1', 'w2', 'w3', 'w4', 'w5'], ['t1', 't2', 't3', 't4']) will yield the result [265.3, [['w1', 't4', 70], ['w2', 't3', 55.3], ['w3', 't2', 95], ['w4', 't1', 45]]].''',
    
    'Solve AP with List': '''useful when you need to solve an Assignment Problem (AP: the goal is to assign a set of tasks to a set of agents in such a way that minimizes the total cost or maximizes the total profit, subject to the limitation that each task is assigned to exactly one agent and each agent is assigned exactly one task.) using lists of agents, tasks, and their associated weights. The input to this tool should include the following parameters:
- goal (str): the goal of the assignment problem, which can be either 'Minimize' or 'Maximize'.
- agents (list): a list of agent identifiers (e.g., ['Agent1', 'Agent1', 'Agent1', 'Agent1', 'Agent2', ...]).
- tasks (list): a list of task identifiers (e.g., ['Task1', 'Task2', 'Task3', 'Task4', 'Task1', ...]).
- weights (list): a list of weights corresponding to the assignment costs or benefits (e.g., [132, 103, 100, 56, 96, ...]).
The output will be a list containing the optimal value and a list of optimal assignments in the format [agent, task, weight].
Example usage: Solve AP with List('Minimize', ['Agent1', 'Agent1', 'Agent1', 'Agent1', 'Agent2', 'Agent2', 'Agent2', 'Agent2', 'Agent3', 'Agent3', 'Agent3', 'Agent3', 'Agent4', 'Agent4', 'Agent4', 'Agent4', 'Agent5', 'Agent5', 'Agent5', 'Agent5', 'Agent6', 'Agent6', 'Agent6', 'Agent6', 'Agent7', 'Agent7', 'Agent7', 'Agent7'], ['Task1', 'Task2', 'Task3', 'Task4', 'Task1', 'Task2', 'Task3', 'Task4', 'Task1', 'Task2', 'Task3', 'Task4', 'Task1', 'Task2', 'Task3', 'Task4', 'Task1', 'Task2', 'Task3', 'Task4', 'Task1', 'Task2', 'Task3', 'Task4', 'Task1', 'Task2', 'Task3', 'Task4'], [132, 103, 100, 56, 96, 90, 102, 109, 96, 92, 122, 60, 91, 74, 123, 91, 92, 119, 157, 92, 86, 108, 92, 46, 115, 159, 101, 105]) will yield the result [312.0, [['Agent1', 'Task3', 100.0], ['Agent4', 'Task2', 74.0], ['Agent5', 'Task1', 92.0], ['Agent6', 'Task4', 46.0]]].''',

    'Solve MCF with List': '''useful when you need to solve a Minimum Cost Flow (MCF: find the most cost-efficient way to send a certain amount of flow through a network, subject to capacity and flow conservation constraints, while minimizing the total transportation cost.) problem given the network's structure. The input to this tool should include the following parameters:
- start_nodes (list): a list of starting nodes for each arc.
- end_nodes (list): a list of ending nodes for each arc.
- capacities (list): a list of capacities for each arc.
- unit_costs (list): a list of unit costs for transporting flow along each arc.
- supplies (list): a list representing the supply/demand at each node (positive for supply, negative for demand).
The output will be a list containing the minimum cost and a list of flow assignments in the format [start_node, end_node, flow, capacity, cost].
Example usage: Solve MCF with List([0, 0, 1, 1, 1, 2, 2, 3, 4], [1, 2, 2, 3, 4, 3, 4, 4, 2], [15, 8, 20, 4, 10, 15, 4, 20, 5], [4, 4, 2, 2, 6, 1, 3, 2, 3], [10, 10, 0, -5, -15]) will yield the result [150, [[0, 1, 12, 15, 48], [0, 2, 8, 8, 32], [1, 2, 8, 20, 16], [1, 3, 4, 4, 8], [1, 4, 0, 10, 0], [2, 3, 12, 15, 12], [2, 4, 4, 4, 12], [3, 4, 11, 20, 22], [4, 2, 0, 5, 0]]].''',

    'Solve MCF with Matrix': '''useful when you need to solve a Minimum Cost Flow (MCF: find the most cost-efficient way to send a certain amount of flow through a network, subject to capacity and flow conservation constraints, while minimizing the total transportation cost.) problem with specified capacities and costs in matrix format. The function takes in:
- capacity_matrix (list of lists): a matrix representing the capacities of the edges.
- cost_matrix (list of lists): a matrix representing the unit costs for the edges.
- supplies (list): a list representing the supply/demand at each node (positive for supply, negative for demand).
The output will be a list containing the minimum cost and a detailed flow assignment in the format [start_node, end_node, flow, capacity, cost].
Example usage: Solve MCF with Matrix([[0, 47, 62, 100, 39, 0], [0, 0, 43, 40, 108, 27], [0, 68, 0, 0, 57, 130], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], [[0, 14, 25, 10, 12, 0], [0, 0, 12, 3, 11, 5], [0, 4, 0, 0, 18, 8], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], [142, 0, 0, -27, -105, -10]) will yield the result [3003, [[0, 1, 47, 47, 658], [0, 2, 29, 62, 725], [0, 3, 27, 100, 270], [0, 4, 39, 39, 468], [1, 2, 0, 43, 0], [1, 3, 0, 40, 0], [1, 4, 66, 108, 726], [1, 5, 0, 27, 0], [2, 1, 19, 68, 76], [2, 4, 0, 57, 0], [2, 5, 10, 130, 80]]].'''

}
# TOOLS_DESCRIPTION.update(TOOLS_DESCRIPTION_PLUS)
# ----------------------------------------------工具函数JSON版---------------------------------------------------
TOOLS_DESCRIPTION_JSON = {
    "Solve TSP with Coordinates":{
    "name": "Solve TSP with Coordinates",
    "description": "useful when you need to solve a Traveling Salesman Problem (TSP: find the shortest possible route that visits each location exactly once and returns to the origin location.) given a set of nodes with specific coordinates.",
    "parameters": {
      "type": "object",
      "properties": {
        "num_nodes": {
          "type": "integer",
          "description": "the number of nodes to visit."
        },
        "coordinates": {
          "type": "object",
          "description": "a dictionary mapping node indices to their (x, y) coordinates, where the keys are node indices and the values are tuples representing the coordinates.",
            "additionalProperties": {
                "type": "array",
                "items": {
                    "type": "number"
                }
            }
        },
        "distance_method": {
          "type": "string",
          "description": "the method to calculate distances, which can be one of the following: \"EUC_2D\", \"MAX_2D\", \"MAN_2D\", \"GEO\" (default is \"EUC_2D\").",
          "enum": ["EUC_2D", "MAX_2D", "MAN_2D", "GEO"]
        }
      },
      "required": ["num_nodes", "coordinates"]
    }
  },
    "Solve TSP with Distance Matrix":{
    "name": "Solve TSP with Distance Matrix",
    "description": "useful when you need to solve a Traveling Salesman Problem (TSP: find the shortest possible route that visits each location exactly once and returns to the origin location.) using a distance matrix representation of the nodes.",
    "parameters": {
      "type": "object",
      "properties": {
        "num_nodes": {
          "type": "integer",
          "description": "the number of nodes to visit."
        },
        "matrix_type": {
          "type": "string",
          "description": "the type of the distance matrix, which can be one of the following: \"LOWER_DIAG_ROW\", \"FULL_MATRIX\", \"UPPER_DIAG_ROW\".",
          "enum": ["LOWER_DIAG_ROW", "FULL_MATRIX", "UPPER_DIAG_ROW"]
        },
        "matrix_data": {
          "type": "array",
          "description": "a list representing the distance values according to the specified matrix type.",
            "items": {
                "type": "array",
                 "items": {
                    "type": "number"
                    }
            }
        }
      },
      "required": ["num_nodes", "matrix_type", "matrix_data"]
    }
  },
    "Solve LP":{
    "name": "Solve LP",
    "description": "useful when you need to solve a Linear Programming (LP: find the best outcome in a model with linear relationships, subject to a set of constraints.) problem defined by an objective function, constraints, and variable bounds.",
    "parameters": {
      "type": "object",
      "properties": {
        "goal": {
          "type": "string",
          "description": "the goal of the LP, which can be either 'Maximize' or 'Minimize'.",
          "enum": ["Maximize", "Minimize"]
        },
        "objective_function": {
          "type": "string",
          "description": "the objective function expressed in lp file format (e.g., 'obj: 8 x + 15 y')."
        },
        "constraints": {
          "type": "array",
          "description": "a list of constraints expressed in lp file format (e.g., ['c1: 10 x + 15 y <= 3000', 'c2: x + y <= 250']).",
          "items": {
            "type": "string"
          }
        },
        "variable_bounds": {
          "type": "array",
          "description": "a list of variable bounds expressed in lp file format (e.g., ['x >= 0', 'y >= 0']).",
            "items": {
                "type": "string"
            }
        }
      },
      "required": ["goal", "objective_function", "constraints", "variable_bounds"]
    }
  },
    "Solve MILP":{
    "name": "Solve MILP",
    "description": "useful when you need to solve a Mixed Integer Linear Programming (MILP: the objective function and constraints are linear, but some of the decision variables are restricted to integer values, while others can be continuous.) problem defined by an objective function, constraints, variable bounds, binary variables, and integer variables.",
    "parameters": {
      "type": "object",
      "properties": {
        "goal": {
          "type": "string",
          "description": "the goal of the MILP, which can be either 'Maximize' or 'Minimize'.",
          "enum": ["Maximize", "Minimize"]
        },
        "objective_function": {
          "type": "string",
          "description": "the objective function expressed in lp file format (e.g., 'obj: 3 x1 + 5 x2 + x3 + x4')."
        },
        "constraints": {
          "type": "array",
          "description": "a list of constraints expressed in lp file format (e.g., ['c1: 2 x1 + x2 + x4 <= 18.5', 'c2: x1 + 2 x2 <= 15', 'c3: x2 + x3 <= 8.5']).",
          "items": {
            "type": "string"
          }
        },
        "variable_bounds": {
          "type": "array",
          "description": "a list of variable bounds expressed in lp file format (e.g., ['x1 >= 0', 'x2 >= 0', 'x3 >= 0']).",
           "items": {
              "type": "string"
           }
        },
        "variable_binaries": {
          "type": "array",
          "description": "a list of binary variables (e.g., ['x4']).",
           "items": {
              "type": "string"
           }
        },
        "variable_integers": {
          "type": "array",
          "description": "a list of integer variables (e.g., ['x1', 'x2']).",
            "items": {
              "type": "string"
           }
        }
      },
      "required": ["goal", "objective_function", "constraints", "variable_bounds", "variable_binaries", "variable_integers"]
    }
  },
    "Solve IP":{
    "name": "Solve IP",
    "description": "useful when you need to solve an Integer Programming (IP: all decision variables are required to take integer values, and both the objective function and constraints are linear.) problem defined by an objective function, constraints, variable bounds, and binary variables.",
    "parameters": {
      "type": "object",
      "properties": {
        "goal": {
          "type": "string",
          "description": "the goal of the IP, which can be either 'Maximize' or 'Minimize'.",
          "enum": ["Maximize", "Minimize"]
        },
        "objective_function": {
          "type": "string",
          "description": "the objective function expressed in lp file format (e.g., 'obj: 8 x1 + 2 x2 + 3 x3')."
        },
        "constraints": {
          "type": "array",
          "description": "a list of constraints expressed in lp file format (e.g., ['c1: 4 x1 + 3 x2 <= 28', 'c2: 3 x1 + 2 x2 <= 19', 'c3: x2 + 3 x3 <= 14']).",
          "items": {
            "type": "string"
          }
        },
         "variable_bounds": {
          "type": "array",
          "description": "a list of variable bounds expressed in lp file format (e.g., ['x1 >= 0', 'x2 >= 0']).",
           "items": {
              "type": "string"
           }
        },
        "variable_binaries": {
          "type": "array",
          "description": "a list of binary variables (e.g., ['x3']).",
           "items": {
              "type": "string"
           }
        }
      },
      "required": ["goal", "objective_function", "constraints", "variable_bounds", "variable_binaries"]
    }
  },
    "Solve MF with List":{
    "name": "Solve MF with List",
    "description": "useful when you need to solve a Maximum Flow (MF: find the greatest possible flow of resources from a source node to a sink node in a network, while respecting the capacities of the edges.) problem using a list representation of edges.",
    "parameters": {
      "type": "object",
      "properties": {
        "start_nodes": {
          "type": "array",
          "description": "a one-dimensional array representing the starting nodes of each edge.",
          "items": {
            "type": "integer"
          }
        },
        "end_nodes": {
          "type": "array",
          "description": "a one-dimensional array representing the ending nodes of each edge.",
          "items": {
            "type": "integer"
          }
        },
        "capacities": {
          "type": "array",
          "description": "a one-dimensional array representing the capacity limits for each edge.",
           "items": {
            "type": "integer"
          }
        }
      },
      "required": ["start_nodes", "end_nodes", "capacities"]
    }
  },
    "Solve MF with Matrix":{
    "name": "Solve MF with Matrix",
    "description": "useful when you need to solve a Maximum Flow (MF: find the greatest possible flow of resources from a source node to a sink node in a network, while respecting the capacities of the edges.) problem using a matrix representation of the capacities.",
    "parameters": {
      "type": "object",
      "properties": {
        "matrix": {
          "type": "array",
          "description": "a 2D array where non-zero values represent the capacity of the edges between nodes. The element at position [i][j] indicates the capacity from node i to node j.",
          "items": {
            "type": "array",
            "items": {
              "type": "integer"
            }
          }
        }
      },
      "required": ["matrix"]
    }
  }
}
TOOLS_DESCRIPTION_PLUS_JSON = {
    "Solve AP with Matrix":{
    "name": "Solve AP with Matrix",
    "description": "useful when you need to solve an Assignment Problem (AP: the goal is to assign a set of tasks to a set of agents in such a way that minimizes the total cost or maximizes the total profit, subject to the limitation that each task is assigned to exactly one agent and each agent is assigned exactly one task.) using a matrix representation of costs or benefits.",
    "parameters": {
      "type": "object",
      "properties": {
        "goal": {
          "type": "string",
          "description": "the goal of the assignment problem, which can be either 'Maximize' or 'Minimize'.",
          "enum": ["Maximize", "Minimize"]
        },
        "matrix_data": {
          "type": "array",
          "description": "a 2D array where each element represents the cost (for minimization) or benefit (for maximization) of assigning agent i to task j.",
          "items": {
            "type": "array",
             "items": {
                 "type": "number"
             }
          }
        },
        "agents": {
          "type": "array",
          "description": "a list of agent identifiers (e.g., ['w1', 'w2', 'w3', 'w4', 'w5']).",
          "items": {
            "type": "string"
          }
        },
        "tasks": {
          "type": "array",
          "description": "a list of task identifiers (e.g., ['t1', 't2', 't3', 't4']).",
            "items": {
                "type": "string"
            }
        }
      },
      "required": ["goal", "matrix_data"]
    }
  },
    "Solve AP with List": {
    "name": "Solve AP with List",
    "description": "useful when you need to solve an Assignment Problem (AP: the goal is to assign a set of tasks to a set of agents in such a way that minimizes the total cost or maximizes the total profit, subject to the limitation that each task is assigned to exactly one agent and each agent is assigned exactly one task.) using lists of agents, tasks, and their associated weights. The input lists must be of the same length, and agents/tasks may contain duplicate values to represent multiple assignment possibilities.",
    "parameters": {
      "type": "object",
      "properties": {
        "goal": {
          "type": "string",
          "description": "The goal of the assignment problem, which can be either 'Maximize' or 'Minimize'.",
          "enum": ["Maximize", "Minimize"]
        },
        "agents": {
          "type": "array",
          "description": "A list of agent identifiers, where each element represents an agent associated with the corresponding task in the 'tasks' list (e.g., ['Agent1', 'Agent1', 'Agent1', 'Agent1', 'Agent2', 'Agent2', 'Agent2', 'Agent2', 'Agent3', 'Agent3', 'Agent3', 'Agent3', 'Agent4', 'Agent4', 'Agent4', 'Agent4']). Agents may appear multiple times to indicate different task assignment options.",
          "items": {
            "type": "string"
          }
        },
        "tasks": {
          "type": "array",
          "description": "A list of task identifiers, where each element represents a task associated with the corresponding agent in the 'agents' list (e.g., ['Task1', 'Task2', 'Task3', 'Task4', 'Task1', 'Task2', 'Task3', 'Task4', 'Task1', 'Task2', 'Task3', 'Task4', 'Task1', 'Task2', 'Task3', 'Task4']). Tasks may appear multiple times to indicate different agent assignment options.",
          "items": {
            "type": "string"
          }
        },
        "weights": {
          "type": "array",
          "description": "A list of numerical weights corresponding to the assignment costs or benefits, where each weight aligns with the agent-task pair at the same index (e.g., [358, 546, 307, 475, 461, 567, 296, 329, 455, 594, 349, 213, 522, 408, 424, 420]). ",
          "items": {
             "type": "number"
           }
        }
      },
      "required": ["goal", "agents", "tasks", "weights"],
    },
    "constraints": [
      "The lists 'agents', 'tasks', and 'weights' must have the same length.",
      "Each element in 'weights' corresponds to an (agent, task) pair at the same index.",
      "The 'agents' and 'tasks' lists may contain duplicates to represent multiple assignment options."
    ]
  },
    "Solve MCF with List":{
    "name": "Solve MCF with List",
    "description": "useful when you need to solve a Minimum Cost Flow (MCF: find the most cost-efficient way to send a certain amount of flow through a network, subject to capacity and flow conservation constraints, while minimizing the total transportation cost.) problem given the network's structure.",
    "parameters": {
      "type": "object",
      "properties": {
        "start_nodes": {
          "type": "array",
          "description": "a list of starting nodes for each arc.",
          "items": {
            "type": "integer"
          }
        },
        "end_nodes": {
          "type": "array",
          "description": "a list of ending nodes for each arc.",
          "items": {
            "type": "integer"
          }
        },
        "capacities": {
          "type": "array",
          "description": "a list of capacities for each arc.",
          "items": {
            "type": "integer"
          }
        },
        "unit_costs": {
          "type": "array",
          "description": "a list of unit costs for transporting flow along each arc.",
          "items": {
            "type": "integer"
          }
        },
        "supplies": {
          "type": "array",
          "description": "a list representing the supply/demand at each node (positive for supply, negative for demand).",
          "items": {
             "type": "integer"
          }
        }
      },
      "required": ["start_nodes", "end_nodes", "capacities", "unit_costs", "supplies"]
    }
  },
    "Solve MCF with Matrix":{
    "name": "Solve MCF with Matrix",
    "description": "useful when you need to solve a Minimum Cost Flow (MCF: find the most cost-efficient way to send a certain amount of flow through a network, subject to capacity and flow conservation constraints, while minimizing the total transportation cost.) problem with specified capacities and costs in matrix format.",
    "parameters": {
      "type": "object",
      "properties": {
        "capacity_matrix": {
          "type": "array",
          "description": "a matrix representing the capacities of the edges.",
            "items": {
              "type": "array",
              "items": {
                "type": "integer"
              }
            }
        },
        "cost_matrix": {
          "type": "array",
          "description": "a matrix representing the unit costs for the edges.",
           "items": {
             "type": "array",
             "items": {
                "type": "integer"
              }
            }
        },
        "supplies": {
          "type": "array",
          "description": "a list representing the supply/demand at each node (positive for supply, negative for demand).",
           "items": {
              "type": "integer"
            }
        }
      },
      "required": ["capacity_matrix", "cost_matrix", "supplies"]
    }
  }
}
# ----------------------------------------------工具名到函数名的映射----------------------------------------------
TOOL2FUNCNAME = {
    'Solve TSP with Coordinates': 'solve_tsp_with_coordinates',
    'Solve TSP with Distance Matrix': 'solve_tsp_with_distance_matrix',
    'Draw Directed Graph': 'draw_directed_graph',
    'Solve LP': 'solve_lp',
    'Solve MILP': 'solve_milp',
    'Solve AP with Matrix': 'solve_ap_with_matrix',
    'Solve AP with List': 'solve_ap_with_list',
    'Solve MF with List': 'solve_mf_with_list',
    'Solve MF with Matrix': 'solve_mf_with_matrix',
   # 'Solve NLP': 'solve_nlp',
    'Solve IP': 'solve_ip',
    'Solve MCF with List': 'solve_mcf_with_list',
    'Solve MCF with Matrix': 'solve_mcf_with_matrix'
}
# ----------------------------------------------需要绘制路线图的表述----------------------------------------------------
route_map_requests = [
    "Please create a route map.",
    "Kindly draw the path map.",
    "Could you plot the route diagram?",
    "Please sketch out the route plan.",
    "Draw a map outlining the route.",
    "Can you illustrate the path on a map?",
    "Please map out the route.",
    "Construct a route diagram, please.",
    "Would you design a map of the route?",
    "Kindly draft the path map.",
    "Please chart the route on a map.",
    "Could you make a map of the route?",
    "Prepare a visual map of the path, please.",
    "Please produce a diagram showing the route.",
    "Draw up a map displaying the route.",
    "Could you lay out the route on a map?",
    "Please depict the path on a map.",
    "Kindly generate a map of the route.",
    "Can you illustrate the route visually?",
    "Please provide a map showing the path.",
    "Please draft a route illustration.",
    "Could you create a path diagram?",
    "Kindly outline the route on a map.",
    "Please generate a route layout.",
    "Could you design a visual representation of the route?",
    "Sketch the route on a map, please.",
    "Kindly produce a diagram of the path.",
    "Please outline the route map.",
    "Can you map out the path, please?",
    "Design a path map, kindly.",
    "Please draw up the route on a diagram.",
    "Could you illustrate the route map?",
    "Kindly draft a diagram of the path.",
    "Could you depict the route visually?",
    "Please illustrate the route on a map.",
    "Can you provide a path diagram?",
    "Kindly map out the path.",
    "Please draft a plan showing the route.",
    "Could you chart the path on a map?",
    "Please provide a layout of the route.",
    "Kindly make a sketch of the route.",
    "Could you create a map showing the path?",
    "Please prepare a diagram for the route.",
    "Could you draw a plan of the route?",
    "Kindly illustrate the path on a map.",
    "Please make a layout of the route.",
    "Could you outline the path diagram?",
    "Please design a map of the path.",
    "Kindly construct a route map.",
    "Could you design a visual map of the route?",
    "Please draw the route on a chart.",
    "Kindly provide a diagram of the path.",
    "Can you sketch the route layout?",
    "Please depict the path on a diagram.",
    "Kindly make a plan for the route.",
    "Could you draw a map depicting the route?",
    "Please illustrate the path layout.",
    "Design a visual route map, kindly.",
    "Can you plot the route on a diagram?",
    "Please sketch the route map.",
    "Could you chart the path on a visual?",
    "Kindly outline the route visually.",
    "Please provide a route representation.",
    "Could you illustrate the path layout?",
    "Kindly map out the route visually.",
    "Could you generate a plan of the route?",
    "Please draw the route diagram.",
    "Kindly design a map for the path.",
    "Could you depict the route on a plan?",
    "Please outline the path in a diagram.",
    "Can you create a map of the route?",
    "Kindly provide a layout of the path.",
    "Please draw a visual of the route.",
    "Could you produce a diagram of the route?",
    "Kindly make a route plan.",
    "Could you draw the path in a diagram?",
    "Please create a map of the path.",
    "Kindly construct a diagram for the route.",
    "Please depict the route in a visual layout.",
    "Could you design a layout for the path?",
    "Please create a diagram showing the route.",
    "Kindly draw up a visual path.",
    "Could you outline a diagram of the route?",
    "Please illustrate the path visually.",
    "Kindly make a layout for the route.",
    "Could you generate a visual route plan?",
    "Please design a route visual representation.",
    "Kindly outline the path map.",
    "Could you depict the route on a chart?",
    "Please draft a route in a diagram.",
    "Can you create a visual path layout?",
    "Kindly produce a diagram showing the path.",
    "Please draw up a map displaying the path.",
    "Could you illustrate a diagram of the route?",
    "Kindly design a route layout visually.",
    "Can you sketch the path map?",
    "Please provide a diagram depicting the route.",
    "Could you make a plan of the path visually?",
    "Kindly draw up the route plan.",
    "Could you chart the path visually on a map?"
]
# ----------------------------------------------线性规划问题无解的表述-------------------------------------------------
lp_without_solution_text = ['All avenues explored have reached dead ends, indicating no viable path forward.',
                       'Despite exhaustive efforts, a solution that satisfies all requirements remains elusive.',
                       'Current constraints create an impasse, rendering any potential solution unfeasible.',
                       'The existing framework offers no wiggle room to accommodate all necessary conditions.',
                       'Reconciling all desired outcomes with the present limitations appears impossible.',
                       'Reaching a point of convergence that fulfills all criteria seems highly improbable.',
                       'The search for a solution has encountered insurmountable obstacles at every turn.',
                       'No amount of adjustment or manipulation can bridge the gap between needs and reality.',
                       'A comprehensive analysis reveals the inherent incompatibility of the given parameters.',
                       'Existing circumstances present an unsolvable puzzle with no apparent resolution.',
                       'All possible permutations have been exhausted, yielding no successful outcome.',
                       'Attempts to find a workable solution have been met with consistent failure.',
                       'The current trajectory inevitably leads to a standstill, with no viable alternative.',
                       'Every potential solution explored has uncovered new and insurmountable obstacles.',
                       'Despite rigorous investigation, no combination of factors produces a satisfactory result.',
                       'A fundamental conflict within the established parameters prevents any viable solution.',
                       'The pursuit of a feasible solution has been stifled by inherent contradictions.',
                       'It is with regret that we must conclude that no suitable solution can be identified.',
                       "Despite our best efforts, we've been unable to devise a plan that meets all requirements.",
                       'Unfortunately, the complexities of this situation preclude any straightforward resolution.',
                       'Our analysis suggests that achieving the desired outcome is simply not possible at this time.',
                       'The intricate web of constraints creates a bottleneck, preventing any progress towards a solution.',
                       'The inherent limitations of the current situation significantly hinder any attempts at a solution.',
                       'There appears to be no realistic scenario where all objectives can be achieved simultaneously.',
                       'The current set of circumstances presents an unsolvable dilemma, hindering further progress.',
                       'The desired outcome remains unattainable given the rigid boundaries of the current context.',
                       'Reaching a compromise that satisfies all parties involved seems like an impossible feat.',
                       'The lack of flexibility within the given parameters severely limits potential solutions.',
                       "After careful consideration, we've determined that no satisfactory solution can be implemented.",
                       'Despite our best efforts to navigate the complexities, no clear path to a solution emerges.',
                       'The interplay of conflicting factors creates a gridlock, halting any progress towards resolution.',
                       'The current parameters present a paradoxical situation where a solution remains perpetually out of reach.',
                       'A comprehensive assessment of all variables points to the absence of a viable solution.',
                       'The inherent nature of the problem itself presents an insurmountable barrier to finding a solution.',
                       'Attempts to reconcile conflicting priorities have proven futile, leading to an impasse.',
                       'The complexities of the situation defy conventional solutions, leaving us at an impasse.',
                       'All known approaches have been explored, revealing no viable path to a successful outcome.',
                       'The existing conditions create a closed loop where any solution attempt leads back to the same constraints.',
                       'A lack of alignment between objectives and realities makes it impossible to arrive at a solution.',
                       'The intricate interplay of factors presents a puzzle with no readily discernible solution.',
                       'The current state of affairs presents a seemingly insurmountable hurdle to achieving the desired result.',
                       'All avenues investigated have led to dead ends, suggesting the absence of a viable solution.',
                       'Attempts to force a solution within the current framework would only exacerbate the existing problems.',
                       'The available options are insufficient to address the full scope of the challenge, leading to a standstill.',
                       'Reconciling all relevant factors within the current context appears to be an impossible task.',
                       'The pursuit of a solution has been hampered by conflicting priorities and insurmountable obstacles.',
                       'The inherent limitations of the situation prevent us from achieving a comprehensive and satisfactory resolution.',
                       'We regret to inform you that, based on our findings, no viable solution can be implemented at this time.',
                       'The search for a solution has uncovered a fundamental incongruity that prevents any forward momentum.',
                       'Despite our meticulous efforts, the constraints of the situation make it impossible to arrive at a workable solution.',
                       'The existing circumstances create a catch-22, where any attempt at a solution is immediately thwarted.',
                       'We have explored all conceivable avenues, yet no feasible path to a solution has presented itself.',
                       'The inherent complexity of the problem, coupled with the existing constraints, renders a solution elusive.',
                       'After extensive analysis, we must conclude that the desired outcome falls outside the realm of possibility.',
                       'The current situation presents a Gordian Knot of conflicting factors, defying any straightforward resolution.',
                       'Our investigation reveals that the desired outcome is incompatible with the fundamental realities of the situation.',
                       'The various elements at play create a stalemate, preventing any meaningful progress towards a solution.',
                       'Despite our best efforts to find a workaround, the limitations of the situation remain insurmountable.',
                       'The existing framework is simply not equipped to accommodate the complexities of the problem at hand.',
                       'We have been forced to conclude that, given the current circumstances, a satisfactory solution is unattainable.',
                       'The inherent contradictions within the problem itself present an insurmountable barrier to finding a solution.',
                       'The desired outcome is contingent upon factors that are beyond our control or influence, rendering it impossible.',
                       'The current parameters create a zero-sum game where any gain in one area necessitates a loss in another.',
                       'Despite exploring all conceivable permutations, no configuration of factors yields a successful outcome.',
                       'The intricate web of interdependencies creates a delicate balance that cannot be disrupted without unintended consequences.',
                       'The current situation resembles a square peg in a round hole, where no amount of force can achieve a proper fit.',
                       'Attempts to impose a solution onto the existing framework would be akin to fitting a square peg into a round hole.',
                       'The pursuit of a solution has been stymied by the immutable laws of logic and the constraints of reality.',
                       'Despite our exhaustive efforts, we have reached an impasse, unable to bridge the gap between aspirations and reality.',
                       'The existing conditions create a paradoxical loop, where any attempted solution only exacerbates the underlying problem.',
                       'The pursuit of a solution has become an exercise in futility, as each attempted path leads to the same dead end.',
                       'The current situation presents a classic case of "damned if you do, damned if you don\'t," with no clear path to success.',
                       'Our analysis has revealed a fundamental flaw within the system itself, rendering any attempt at a solution futile.',
                       'The conflicting demands and limited resources create an intractable problem with no readily apparent solution.',
                       'The search for a solution has uncovered a fundamental incompatibility between the desired outcome and the existing reality.',
                       'The current state of affairs presents a formidable challenge that defies conventional solutions and necessitates a radical rethink.',
                       'Despite our best efforts to find common ground, the divergent priorities of the involved parties make a solution impossible.',
                       'The complexities of the situation have created a quagmire, trapping us in a cycle of unsolvable problems and unintended consequences.',
                       'The pursuit of a solution has become an exercise in frustration, as every step forward seems to be met with two steps back.',
                       'We are faced with a situation where any attempted solution would necessitate a cascade of compromises, ultimately undermining the desired outcome.',
                       'The search for a solution has been hampered by a lack of clear information, conflicting data, and a general ambiguity surrounding the problem itself.',
                       'The existing framework, while well-intentioned, is simply not robust enough to accommodate the complexities of the situation and provide a viable solution.',
                       'The desired outcome, while admirable in theory, proves to be impractical and unattainable within the confines of the current circumstances.',
                       'Despite our commitment to finding a resolution, the reality of the situation dictates that a satisfactory solution remains elusive and beyond our grasp.',
                       'The conflicting priorities, coupled with the inherent limitations of the system, create an unsolvable puzzle that defies even the most creative solutions.',
                       'We have exhausted all available resources and explored every conceivable avenue, yet the problem persists, stubbornly resistant to any attempts at resolution.',
                       'The current situation resembles a game of chess where every possible move results in checkmate, leaving us with no strategic options for success.',
                       'The inherent contradictions within the problem itself create a paradoxical loop, rendering any attempt at a solution self-defeating and ultimately futile.',
                       'The complexities of the situation, compounded by unforeseen circumstances and external factors, have conspired to create an intractable problem with no foreseeable solution.',
                       'Despite our unwavering determination, the search for a solution has become a Sisyphean task, with each attempted solution inevitably unraveling and leading back to the starting point.',
                       'The conflicting interests of the various stakeholders create a stalemate, rendering any attempt at a compromise or negotiated settlement impossible.',
                       "The current situation resembles a Rubik's Cube where every twist and turn only seems to complicate the puzzle further, pushing us further away from a solution.",
                       'The pursuit of a solution has been characterized by a series of false starts, dead ends, and unforeseen obstacles, ultimately leading to an impasse.',
                       'The existing framework, while seemingly logical and well-structured, suffers from fundamental flaws that prevent it from effectively addressing the complexities of the situation.',
                       'The desired outcome, while desirable in principle, proves to be unattainable in practice due to the intractable nature of the problem and the limitations of our current capabilities.',
                       'The search for a solution has been hampered by a lack of precedent, leaving us to navigate uncharted territory without a roadmap or compass to guide our way.',
                       'The existing conditions present a classic example of a wicked problem, characterized by complexity, uncertainty, and a lack of clear solutions.',
                       'Despite our best efforts to approach the problem from multiple angles and leverage diverse perspectives, the solution remains stubbornly elusive.',
                       'The inherent ambiguity of the situation, coupled with a lack of reliable data and conflicting information, makes it impossible to arrive at a definitive and actionable solution.',
                       'We are forced to acknowledge that the problem, in its current form, is simply not solvable, requiring a fundamental shift in perspective or a complete reframing of the issue itself.']
# print(f'{TOOLS_DESCRIPTION_PLUS_JSON}')