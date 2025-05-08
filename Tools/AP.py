from ortools.linear_solver import pywraplp
import pandas as pd
import numpy as np
from ortools.sat.python import cp_model


def solve_ap_with_matrix(goal, matrix_data, agents=None, tasks=None):
    assert goal in ['Maximize', 'Minimize']
    matrix_data = [[elem for elem in row if elem != 0] for row in matrix_data]
    num_agents = len(matrix_data)
    num_tasks = len(matrix_data[0])
    assert num_agents >= num_tasks
    # if num_agents < num_tasks:
    #     matrix_data = np.array(matrix_data).T.tolist()

    solver = pywraplp.Solver.CreateSolver("SCIP")

    if not solver:
        return None, None


    x = {}
    for i in range(num_agents):
        for j in range(num_tasks):
            x[i, j] = solver.IntVar(0, 1, "")


    for i in range(num_agents):
        solver.Add(solver.Sum([x[i, j] for j in range(num_tasks)]) <= 1)

    for j in range(num_tasks):
        solver.Add(solver.Sum([x[i, j] for i in range(num_agents)]) == 1)

    objective_terms = []
    for i in range(num_agents):
        for j in range(num_tasks):
            objective_terms.append(matrix_data[i][j] * x[i, j])
    if goal == 'Minimize':
        solver.Minimize(solver.Sum(objective_terms))
    else:
        solver.Maximize(solver.Sum(objective_terms))
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        optimal_value = solver.Objective().Value()
        optimal_assignment = []
        for i in range(num_agents):
            for j in range(num_tasks):
                if x[i, j].solution_value() > 0.5:
                    optimal_assignment.append([i, j, matrix_data[i][j]]) if agents is None and tasks is None else optimal_assignment.append([agents[i], tasks[j], matrix_data[i][j]])
        return [optimal_value, optimal_assignment]
    else:
        return [None, None]


# 示例使用
# solve_ap_with_matrix("Minimize", [
#     [90.5, 80, 75, 70],
#     [35, 85, 55.3, 65],
#     [125, 95, 90, 95],
#     [45, 110.2, 95, 115],
#     [50, 100, 90, 100],
# ],['w1','w2','w3','w4','w5'],['t1','t2','t3','t4'])
# 运行结果
# [265.3, [['w1', 't4', 70], ['w2', 't3', 55.3], ['w3', 't2', 95], ['w4', 't1', 45]]]

def solve_ap_with_list(goal, agents, tasks, weights):
    assert goal in ['Minimize', 'Maximize']
    assert len(agents) == len(tasks) == len(weights)
    assert isinstance(weights, list)
    assert isinstance(agents, list)
    assert isinstance(tasks, list)
    data = {
        'agent': agents,
        'task': tasks,
        'weight': [float(w) for w in weights]
    }
    data = pd.DataFrame(data)

    model = cp_model.CpModel()

    x = model.new_bool_var_series(name="x", index=data.index)


    for unused_name, tasks in data.groupby("agent"):
        model.add_at_most_one(x[tasks.index])

    for unused_name, workers in data.groupby("task"):
        model.add_exactly_one(x[workers.index])

    if goal == 'Minimize':
        model.minimize(data.weight.dot(x))
    else:
        model.maximize(data.weight.dot(x))

    solver = cp_model.CpSolver()
    status = solver.solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        optimal_value = solver.ObjectiveValue()
        optimal_assignment = []
        selected = data.loc[solver.boolean_values(x).loc[lambda x: x].index]
        for unused_index, row in selected.iterrows():
            optimal_assignment.append([row.agent, row.task, row.weight])
        return [optimal_value, optimal_assignment]
    elif status == cp_model.INFEASIBLE:
        return [None, None]
    else:
        return [None, None]
    
# print(solve_ap_with_matrix()
# 示例使用
# solve_ap_with_list('Minimize', ['Agent1', 'Agent1', 'Agent1', 'Agent1', 'Agent2', 'Agent2', 'Agent2', 'Agent2', 'Agent3', 'Agent3', 'Agent3', 'Agent3', 'Agent4', 'Agent4', 'Agent4', 'Agent4', 'Agent5', 'Agent5', 'Agent5', 'Agent5', 'Agent6', 'Agent6', 'Agent6', 'Agent6', 'Agent7', 'Agent7', 'Agent7', 'Agent7'], ['Task1', 'Task2', 'Task3', 'Task4', 'Task1', 'Task2', 'Task3', 'Task4', 'Task1', 'Task2', 'Task3', 'Task4', 'Task1', 'Task2', 'Task3', 'Task4', 'Task1', 'Task2', 'Task3', 'Task4', 'Task1', 'Task2', 'Task3', 'Task4', 'Task1', 'Task2', 'Task3', 'Task4'], [132, 103, 100, 56, 96, 90, 102, 109, 96, 92, 122, 60, 91, 74, 123, 91, 92, 119, 157, 92, 86, 108, 92, 46, 115, 159, 101, 105])
# 运行结果
# [312.0, [['Agent1', 'Task3', 100.0], ['Agent4', 'Task2', 74.0], ['Agent5', 'Task1', 92.0], ['Agent6', 'Task4', 46.0]]]

# def solve_ap_from_csv(goal, file_path):
#     assert goal in ['Minimize', 'Maximize']
#     assert file_path.endswith('.csv')
#     # 读取CSV文件
#     # 第一列为agent 第二列为work 第三列为cost或benefit
#     data = pd.read_csv(file_path)

#     # Model
#     model = cp_model.CpModel()

#     # Variables
#     x = model.new_bool_var_series(name="x", index=data.index)

#     # Constraints
#     # Each worker is assigned to at most one task.
#     for unused_name, tasks in data.groupby("worker"):
#         model.add_at_most_one(x[tasks.index])

#     # Each task is assigned to exactly one worker.
#     for unused_name, workers in data.groupby("task"):
#         model.add_exactly_one(x[workers.index])

#     # Objective
#     if goal == 'Minimize':
#         model.minimize(data.cost.dot(x))
#     else:
#         model.maximize(data.cost.dot(x))

#     # Solve
#     solver = cp_model.CpSolver()
#     status = solver.solve(model)

#     # Print solution.
#     if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
#         optimal_value = solver.ObjectiveValue()
#         optimal_assignment = []
#         # print(f"Total cost = {solver.objective_value}\n")
#         selected = data.loc[solver.boolean_values(x).loc[lambda x: x].index]
#         for unused_index, row in selected.iterrows():
#             optimal_assignment.append([row.worker, row.task, row.cost])
#             # print(f"{row.task} assigned to {row.worker} with a cost of {row.cost}")
#         return [optimal_value, optimal_assignment]
#     elif status == cp_model.INFEASIBLE:
#         print("No solution found")
#         return [None, None]
#     else:
#         print("Something is wrong, check the status and the log of the solve")
#         return [None, None]

# 示例使用
# solve_ap_from_csv("Maximize", r"/home/zjl/OR/Code/RUN/Tools/ap.csv")
# 运行结果
# [425.0, [['w1', 't3', 75], ['w3', 't1', 125], ['w4', 't4', 115], ['w5', 't2', 110]]]


# def solve_ap_from_excel(goal, file_path):
#     assert goal in ['Minimize', 'Maximize']
#     assert file_path.endswith('.xlsx')
#     # 读取excel文件
#     # 第一列为agent 第二列为work 第三列为cost或benefit
#     data = pd.read_excel(file_path)

#     # Model
#     model = cp_model.CpModel()

#     # Variables
#     x = model.new_bool_var_series(name="x", index=data.index)

#     # Constraints
#     # Each worker is assigned to at most one task.
#     for unused_name, tasks in data.groupby("worker"):
#         model.add_at_most_one(x[tasks.index])

#     # Each task is assigned to exactly one worker.
#     for unused_name, workers in data.groupby("task"):
#         model.add_exactly_one(x[workers.index])

#     # Objective
#     if goal == 'Minimize':
#         model.minimize(data.cost.dot(x))
#     else:
#         model.maximize(data.cost.dot(x))

#     # Solve
#     solver = cp_model.CpSolver()
#     status = solver.solve(model)

#     # Print solution.
#     if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
#         optimal_value = solver.ObjectiveValue()
#         optimal_assignment = []
#         # print(f"Total cost = {solver.objective_value}\n")
#         selected = data.loc[solver.boolean_values(x).loc[lambda x: x].index]
#         for unused_index, row in selected.iterrows():
#             optimal_assignment.append([row.worker, row.task, row.cost])
#             # print(f"{row.task} assigned to {row.worker} with a cost of {row.cost}")
#         return [optimal_value, optimal_assignment]
#     elif status == cp_model.INFEASIBLE:
#         print("No solution found")
#         return [None, None]
#     else:
#         print("Something is wrong, check the status and the log of the solve")
#         return [None, None]
# # 示例使用
# # solve_ap_from_excel("Maximize", r"/home/zjl/OR/Code/RUN/Tools/ap.xlsx")
# # 运行结果
# # [425.0, [['w1', 't3', 75], ['w3', 't1', 125], ['w4', 't4', 115], ['w5', 't2', 110]]]

