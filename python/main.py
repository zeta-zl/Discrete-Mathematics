import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# 传入一个邻接矩阵，返回一个邻接矩阵
def adjacency_matrix(input_data):
    # 将输入转换为列表
    input_list = input_data.strip().split('\n')

    # 获取图中的节点数
    num_nodes = len(input_list)

    # 创建一个num_nodes x num_nodes的矩阵，所有值都为0
    matrix = [[0] * num_nodes for _ in range(num_nodes)]

    # 填充矩阵
    for i in range(num_nodes):
        for j, value in enumerate(input_list[i].split()):
            matrix[i][j] = int(value)

    return matrix


# 输入一个邻接矩阵，返回一个邻接矩阵
def matrix_to_matrix():
    # 从键盘接受用户输入
    input_data = ''
    while True:
        line = input()
        if line:
            input_data += line + '\n'
        else:
            break

    # 调用adjacency_matrix函数返回邻接矩阵
    mat = adjacency_matrix(input_data)

    return mat


# 传入一个邻接表，返回一个邻接矩阵
def adjacency_list(input_data):
    # 生成邻接表的字典
    temp_adjacency_list = {i: list(map(int, line.split()[1:])) for i, line in enumerate(input_data.strip().split('\n'))}
    # 生成一个nxn矩阵
    num_nodes = len(temp_adjacency_list)
    matrix = [[0] * num_nodes for _ in range(num_nodes)]
    # 对矩阵赋值
    for i, neighbors in temp_adjacency_list.items():
        for j in neighbors:
            matrix[i][j - 1] = 1
    return matrix


# 输入一个邻接表，返回一个邻接矩阵
def list_to_matrix():
    # 从键盘接受用户输入
    input_data = ''
    while True:
        line = input()
        if line:
            input_data += line + '\n'
        else:
            break

    ret = adjacency_list(input_data)

    return ret


# 传入边集，返回邻接矩阵
def edge_list(input_data):  # 边集转换为邻接矩阵
    input_data = [tuple(map(int, edge.split())) for edge in input_data.split("\n") if edge]
    vertex_list = set([vertex for edge in input_data for vertex in edge])
    n = max(vertex_list)
    adj_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for edge in input_data:
        adj_matrix[edge[0] - 1][edge[1] - 1] = 1
    return adj_matrix


# 输入边集，返回邻接矩阵
def edge_to_matrix():
    # 从键盘接受用户输入
    input_data = ''
    while True:
        line = input()
        if line:
            input_data += line + '\n'
        else:
            break

    # 调用edge_list函数返回邻接矩阵
    mat = edge_list(input_data)

    return mat


# 将邻接矩阵转换为图对象
def matrix_to_graph(matrix):
    # 将矩阵转换为numpy矩阵
    np_matrix = np.matrix(matrix)

    # 使用from_numpy_matrix函数返回图对象
    G = nx.from_numpy_matrix(np_matrix)

    return G


# 创建一个带权图
def graph_with_weight():
    """
    从键盘输入若干个三元组，并返回生成的图对象。
    """
    # 创建一个带权图
    G = nx.Graph()

    # 循环输入三元组
    while True:
        # 从键盘输入一行文本
        line = input().strip()

        # 如果输入为空，结束循环
        if not line:
            break

        # 解析三元组
        a, b, c = line.split()

        # 添加边
        G.add_edge(a, b, weight=int(c))

    return G


# 接受输入，输入1，2或3表示输入邻接矩阵，邻接表或是边集
# 仅接受无权图

# 输入格式示例：
# 邻接矩阵：
# 0 1 1
# 1 0 1
# 1 1 0
# 邻接表：
# 1 2 3
# 2 3
# （表示1与2，3相连，2与3相连）
# 边集：
# 1 2
# 2 3
# 1 4
# （表示1与2，2与3，1与4相连）
def get_input():
    # 获取用户输入的类型
    input_type = int(input())

    # 调用对应的函数将输入转换为邻接矩阵
    if input_type == 1:
        matrix = matrix_to_matrix()
    elif input_type == 2:
        matrix = list_to_matrix()
    elif input_type == 3:
        matrix = edge_to_matrix()
    else:
        print("Invalid input type.")
        return None

    return matrix


# 绘制图
def plot_graph(G, color=False, pos=None):
    """
      绘制给定的图像。

      参数：
      G -- 一个networkx中的图对象
      color -- 可选参数，如果设置为True，则染色
      pos -- 可选参数，表示节点位置的字典
      """
    # 使用pos参数绘制点
    if pos is None:
        pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, with_labels=True)

    # 如果需要，染色
    if color:
        colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'purple'}
        Colors = nx.greedy_color(G)
        nx.draw_networkx_nodes(G, pos, node_color=[colors[Colors[n]] for n in G.nodes()])

    plt.show()


# 绘制带权图
def show_weighted_graph(G):
    """
    绘制带权图。
    """
    # 计算点的位置
    pos = nx.spring_layout(G)

    # 绘制边
    nx.draw(G, pos, with_labels=True, node_size=300)

    # 添加边权标签
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # 显示图像
    plt.show()


def minimum_spanning_tree(G):
    """
    返回最小生成树。

    参数：
    G -- 一个networkx中的图对象

    返回值：
    一个networkx中的最小生成树图对象
    """
    # 计算最小生成树
    mst = nx.minimum_spanning_tree(G, algorithm="kruskal")

    # 返回最小生成树
    return mst


# 获得两点间最短路径
def shortest_path(G, source, target):
    """
    返回从源点到目标点的最短路径以及经过的路径。

    参数：
    G -- 一个networkx中的图对象
    source -- 源点
    target -- 目标点

    返回值：
    一个包含最短路径长度和路径节点的元组
    """
    # 计算最短路径长度
    length = nx.shortest_path_length(G, source, target)

    # 计算最短路径
    path = nx.shortest_path(G, source, target)

    # 返回最短路径长度和路径节点
    return length, path


# 获得一个图的补图
def complement(G):
    """
    返回给定图的补图。

    参数：
    G -- 一个networkx中的图对象

    返回值：
    一个networkx中的补图
    """
    # 计算补图
    H = nx.complement(G)

    # 返回补图
    return H


# 计算欧拉回路
def eulerian_circuit(G, start=None):
    """
    返回给定图的欧拉回路。

    参数：
    G -- 一个networkx中的图对象
    start -- 起点（可选）

    返回值：
    一个包含欧拉回路边的迭代器
    """
    # 计算欧拉回路
    circuit = nx.eulerian_circuit(G, start)

    # 返回欧拉回路
    return circuit


# 计算哈密顿回路
def hamiltonian_circuit(G, start=None):
    """
    返回给定图的哈密顿回路。

    参数：
    G -- 一个networkx中的图对象
    start -- 起点（可选）

    返回值：
    一个包含哈密顿回路边的迭代器
    """
    # 计算哈密顿回路
    circuit = nx.hamiltonian_circuit(G, start)

    # 返回哈密顿回路
    return circuit


# 作出有向图
def draw_directed_graph(edges, pos=None):
    """
    画出给定边构成的有向图。

    参数：
    edges -- 一个包含边的迭代器
    pos -- 点的位置（可选）
    """
    # 创建有向图
    G = nx.DiGraph()

    # 将边添加到图中
    G.add_edges_from(edges)

    # 确定点的位置
    pos = nx.kamada_kawai_layout(G)
    # 画出图像，带有箭头
    nx.draw_networkx_edges(G, pos, arrowstyle='->')

    nx.draw(G, pos=pos, with_labels=True)

    # 显示图像
    plt.show()


# 标注出一个图的子图
def show_subgraph(G, subG):
    """
    绘制图G，并用红色标注出子图subG。
    """
    # 计算点的位置
    pos = nx.spring_layout(G)

    # 绘制边
    nx.draw(G, pos, with_labels=True, node_size=300)

    # 绘制子图的边
    nx.draw(subG, pos, with_labels=True, node_size=300, edge_color='red')

    # 显示图像
    plt.show()


# 在给定图中找到V中任意两点的最短路径的最小值。
def min_shortest_path(G, V):
    """
    在给定图中找到V中任意两点的最短路径的最小值。

    参数：
    G -- 一个NetworkX图对象
    V -- G中的顶点列表

    返回值：
    一个包含起点，终点和元组列表表示的最短路径边的三元组
    """
    min_length = float('inf')
    min_path = (None, None, [])
    for u in V:
        for v in V:
            if u != v:
                length, path = nx.single_source_dijkstra(G, u, v, weight='weight')
                if length < min_length:
                    min_length = length
                    min_path = (u, v, [(path[i], path[i + 1]) for i in range(len(path) - 1)])
    return min_path


# 根据度序列生成一个图。
def generate_graph(degree_sequence):
    """
    根据度序列生成一个图。

    参数：
    degree_sequence -- 度序列（一个整数列表）

    返回值：
    生成的图（一个NetworkX图对象）
    """
    # 根据度序列生成图
    try:
        G = nx.configuration_model(degree_sequence)
    except nx.NetworkXError:
        print("不是有效的度数列")
        return None
    return G


# 根据度序列生成一个简单图。
def generate_simple_graph(degree_sequence):
    """
    根据度序列生成一个简单图。

    参数：
    degree_sequence -- 度序列（一个整数列表）

    返回值：
    生成的图（一个NetworkX图对象）
    """
    # 根据度序列生成图
    try:
        G = nx.havel_hakimi_graph(degree_sequence)
    except nx.NetworkXError:
        print("不是有效的度数列")
        return None
    return G


# 将带权图转换成无权图
def drop_weights(G):
    """
    Drop the weights from a networkx weighted graph.
    """
    for node, edges in nx.to_dict_of_dicts(G).items():
        for edge, attrs in edges.items():
            attrs.pop('weight', None)


# 判断图是否是平面图，返回平面图的坐标
def get_coordinates(G):
    if nx.check_planarity(G)[0]:
        # 如果是平面图，使用planar_layout函数获取坐标
        pos = nx.planar_layout(G)
        return pos
    else:
        # 如果不是平面图，返回None
        return None


def operator():
    degree = [3, 3, 3, 3, 3, 3]
    G = generate_simple_graph(degree)  # 通过度序列得到简单图
    pos = get_coordinates(G)
    plot_graph(G, pos=pos)


if __name__ == "__main__":
    operator()
    # try:
    #     operator()
    # except TypeError:
    #     print("Invalid input format.")
    # except ValueError:
    #     print("Invalid input value.")
    # except nx.NetworkxError:
    #     print("图错误，请检查输入的图是否具有相关性质")
    # except Exception:
    #     print("An unknown error occurred.")
