import numpy as np
import pandas as pd
from ortools.graph.python import max_flow

def solve_mf_with_list(start_nodes, end_nodes, capacities):
    smf = max_flow.SimpleMaxFlow()

    all_arcs = smf.add_arcs_with_capacity(start_nodes, end_nodes, capacities)

    status = smf.solve(min(start_nodes), max(end_nodes))

    if status != smf.OPTIMAL:
        return [None,None]

    max_flow_value = smf.optimal_flow()

    solution_flows = smf.flows(all_arcs)
    flow_solution = []
    for arc, flow, capacity in zip(all_arcs, solution_flows, capacities):
        flow_solution.append([smf.tail(arc), smf.head(arc), int(flow), int(capacity)])

    return [max_flow_value, flow_solution]

# 示例使用
# solve_max_flow_with_list([0, 0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 3, 2, 4, 3, 4, 2, 4], [20, 30, 10, 40, 30, 10, 20, 5, 20])
# 运行结果
# [60, [[0, 1, 20, 20], [0, 2, 30, 30], [0, 3, 10, 10], [1, 2, 0, 40], [1, 4, 20, 30], [2, 3, 10, 10], [2, 4, 20, 20], [3, 2, 0, 5], [3, 4, 20, 20]]]
def solve_mf_with_matrix(matrix):
    """Solve the given max flow problem."""
    # Instantiate a SimpleMaxFlow solver.
    start_nodes = []
    end_nodes = []
    capacities = []
    try:
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] != 0:  # 假设非零值表示有边
                    start_nodes.append(i)
                    end_nodes.append(j)
                    capacities.append(matrix[i][j])
        start_nodes = np.array(start_nodes)
        end_nodes = np.array(end_nodes)
        capacities = np.array(capacities)
        return solve_mf_with_list(start_nodes, end_nodes, capacities)
    except:
        return [None,None]
# 示例使用
# solve_max_flow_with_matrix([[0.0, 8.0, 10.0, 1.0, 8.0, 2.0],
#  [0.0, 0.0, 10.0, 2.0, 0.0, 6.0],
#  [0.0, 0.0, 0.0, 7.0, 8.0, 6.0],
#  [0.0, 10.0, 6.0, 0.0, 2.0, 10.0],
#  [0.0, 0.0, 4.0, 3.0, 0.0, 3.0],
#  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
# 运行结果
# 60, [[0, 1, 20, 20], [0, 2, 30, 30], [0, 3, 10, 10], [1, 2, 0, 40], [1, 4, 20, 30], [2, 3, 10, 10], [2, 4, 20, 20], [3, 2, 0, 5], [3, 4, 20, 20]]

# def solve_max_flow_from_csv(file_path):
#     df = pd.read_csv(file_path)
#     start_nodes = df['start_nodes'].tolist()
#     end_nodes = df['end_nodes'].tolist()
#     capacities = df['capacity'].tolist()
#     return solve_max_flow_with_list(start_nodes, end_nodes, capacities)
# 示例使用
# solve_max_flow_from_csv('mf.csv')
# 运行结果
# [60, [[0, 1, 20, 20], [0, 2, 30, 30], [0, 3, 10, 10], [1, 2, 0, 40], [1, 4, 20, 30], [2, 3, 10, 10], [2, 4, 20, 20], [3, 2, 0, 5], [3, 4, 20, 20]]]

# def solve_max_flow_from_excel(file_path):
#     df = pd.read_excel(file_path)
#     start_nodes = df['start_nodes'].tolist()
#     end_nodes = df['end_nodes'].tolist()
#     capacities = df['capacity'].tolist()
#     return solve_max_flow_with_list(start_nodes, end_nodes, capacities)
# 示例使用
# solve_max_flow_from_excel('mf.xlsx')
# 运行结果
# [60, [[0, 1, 20, 20], [0, 2, 30, 30], [0, 3, 10, 10], [1, 2, 0, 40], [1, 4, 20, 30], [2, 3, 10, 10], [2, 4, 20, 20], [3, 2, 0, 5], [3, 4, 20, 20]]]
