import numpy as np

from ortools.graph.python import min_cost_flow

def solve_mcf_with_list(start_nodes, end_nodes, capacities, unit_costs,supplies):
    assert len(start_nodes) == len(end_nodes) == len(capacities) == len(unit_costs)
    assert isinstance(start_nodes, list)
    assert isinstance(end_nodes, list)
    assert isinstance(capacities, list)
    assert isinstance(unit_costs, list)
    assert isinstance(supplies, list)
    assert sum(supplies) == 0
    assert (max(end_nodes)+1) == len(supplies)
    smcf = min_cost_flow.SimpleMinCostFlow()
    all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(
        start_nodes, end_nodes, capacities, unit_costs
    )

    smcf.set_nodes_supplies(np.arange(0, len(supplies)), supplies)

    status = smcf.solve()

    if status != smcf.OPTIMAL:
        return [None,None]
    min_cost_value = smcf.optimal_cost()
    # print(f"Minimum cost: {smcf.optimal_cost()}")
    # print(" Arc    Flow / Capacity Cost")
    # 获取分配
    solution_flows = smcf.flows(all_arcs)
    costs = solution_flows * unit_costs
    flow_solution = []
    for arc, flow, cost in zip(all_arcs, solution_flows, costs):
        flow_solution.append([smcf.tail(arc), smcf.head(arc), int(flow), smcf.capacity(arc), int(cost)])
    #     print(
    #         f"{smcf.tail(arc):1} -> {smcf.head(arc)}  {flow:3}  / {smcf.capacity(arc):3}       {cost}"
    #     )
    # print(min_cost_value,flow_solution)
    return [min_cost_value,flow_solution]
# 示例使用
# solve_mcf_with_list([0, 0, 1, 1, 1, 2, 2, 3, 4],[1, 2, 2, 3, 4, 3, 4, 4, 2],[15, 8, 20, 4, 10, 15, 4, 20, 5],[4, 4, 2, 2, 6, 1, 3, 2, 3],[10, 10, 0, -5, -15])
# 运行结果
# 150 [[0, 1, 12, 15, 48], [0, 2, 8, 8, 32], [1, 2, 8, 20, 16], [1, 3, 4, 4, 8], [1, 4, 0, 10, 0], [2, 3, 12, 15, 12], [2, 4, 4, 4, 12], [3, 4, 11, 20, 22], [4, 2, 0, 5, 0]]

def solve_mcf_with_matrix(capacity_matrix, cost_matrix, supplies):
    start_nodes = []
    end_nodes = []
    capacities = []
    unit_costs = []
    for i in range(len(capacity_matrix)):
        for j in range(len(capacity_matrix[i])):
            if capacity_matrix[i][j] != 0:  # 假设非零值表示有边
                start_nodes.append(i)
                end_nodes.append(j)
                capacities.append(capacity_matrix[i][j])
    for i in range(len(cost_matrix)):
        for j in range(len(cost_matrix[i])):
            if cost_matrix[i][j] != 0:  # 假设非零值表示有边
                unit_costs.append(cost_matrix[i][j])
    return solve_mcf_with_list(start_nodes, end_nodes, capacities, unit_costs,supplies)
# 示例使用
# solve_mcf_with_matrix([[0, 47, 62, 100, 39, 0], [0, 0, 43, 40, 108, 27], [0, 68, 0, 0, 57, 130], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], [[0, 14, 25, 10, 12, 0], [0, 0, 12, 3, 11, 5], [0, 4, 0, 0, 18, 8], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], [142, 0, 0, -27, -105, -10])
# 运行结果
# [3003, [[0, 1, 47, 47, 658], [0, 2, 29, 62, 725], [0, 3, 27, 100, 270], [0, 4, 39, 39, 468], [1, 2, 0, 43, 0], [1, 3, 0, 40, 0], [1, 4, 66, 108, 726], [1, 5, 0, 27, 0], [2, 1, 19, 68, 76], [2, 4, 0, 57, 0], [2, 5, 10, 130, 80]]]
# print(solve_mcf_with_list(goal='Minimize', agents=['Depot 0', 'Depot 1', 'Depot 2', 'Site 3', 'Site 4', 'Site 5'], tasks=['Depot 0 to Depot 1', 'Depot 0 to Depot 2', 'Depot 0 to Site 3', 'Depot 1 to Site 4', 'Depot 1 to Site 5', 'Depot 2 to Site 4'], weights=[22, 97, 52, 79, 93, 118]))
# print(solve_mcf_with_matrix(capacity_matrix=[[0, 858, 0, 287, 875, 0, 0, 133], [0, 0, 0, 0, 0, 149, 0, 844], [0, 0, 0, 0, 0, 0, 196, 486], [0, 0, 0, 0, 700, 0, 99, 860], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], cost_matrix=[[0, 114, 0, 135, 115, 0, 0, 88], [0, 0, 0, 0, 0, 163, 0, 178], [0, 0, 0, 0, 0, 0, 176, 180], [0, 0, 0, 0, 146, 0, 104, 68], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], supplies=[61, 0, 0, 0, -47, -2, -4, -8]))