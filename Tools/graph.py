import matplotlib.pyplot as plt
import networkx as nx
# ----------------------------------------------绘制有向图----------------------------------------------
def draw_directed_graph(edges, node_coords=None):
    G = nx.DiGraph()
    # 添加有向边及权重
    for edge in edges:
        start, end, weight = edge
        if weight != 0:
            G.add_edge(start, end, weight=weight)

    # 如果提供了节点坐标，则使用它们绘制节点
    if node_coords:
        pos = {node: (coord[0], coord[1]) for node, coord in node_coords.items()}
        G.add_nodes_from(pos.keys())
    else:
        # 如果没有提供节点坐标，自动生成布局
        pos = nx.circular_layout(G)

    # 绘制图形
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold', arrowsize=15)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    # 自定义标签
    # edge_labels = {(u, v): f"{w}个" for (u, v, w) in edge_list}
    # 添加单位到权重标签
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # 打印图片的绝对路径
    image_path = f'directed_graph_{len(edges)}.png'
    
    # 图片保存
    plt.savefig(image_path, format="png", dpi=300)
    return image_path
# 示例使用1 TSP无坐标
# draw_directed_graph([[1, 3, 24],
#   [3, 5, 18],
#   [5, 2, 11],
#   [2, 4, 13],
#   [4, 1, 32]])
# 运行结果1
'directed_graph_10.png'
# 示例使用2 TSP有坐标
# draw_directed_graph([[1, 3, 24],
#   [3, 5, 18],
#   [5, 2, 11],
#   [2, 4, 13],
#   [4, 1, 32]],{1: (16, 18), 2: (23, 46), 3: (36, 24), 4: (23, 53), 5: (17, 34)})
# 运行结果2
# 'directed_graph_5.png'
# 示例使用3 最大流
# print(draw_directed_graph([[0, 1, 40],[0,2,30],[0,3,0],[1,2,20],[1,4,20],[2,3,20],[2,4,20],[3,2,0],[3,4,20]]))