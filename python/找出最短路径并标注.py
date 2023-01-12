import networkx as nx
import matplotlib.pyplot as plt


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


def convert_to_tuples(lst):
    """
    将一个整数列表转换为元组列表。

    参数：
    lst -- 一个整数列表

    返回值：
    一个元组列表，其中每个元组都是输入列表中的连续元素对
    """
    return [(lst[i], lst[i + 1]) for i in range(len(lst) - 1)]


# Create a graph using the Graph class
G = nx.Graph()

# Add edges to the graph, with weights
G.add_edge(1, 2, weight=2.0)
G.add_edge(1, 3, weight=3.0)
G.add_edge(1, 4, weight=4.0)
G.add_edge(2, 3, weight=5.0)
G.add_edge(2, 5, weight=6.0)
G.add_edge(3, 4, weight=7.0)
G.add_edge(3, 6, weight=8.0)
G.add_edge(4, 5, weight=9.0)
G.add_edge(4, 6, weight=10.0)
G.add_edge(5, 6, weight=11.0)

# 获得布局信息
pos = nx.spring_layout(G)

# 将边权信息添加进图中
for u, v, weight in G.edges(data='weight'):
    G[u][v]['length'] = weight

# 作出图像
nx.draw(G, pos=pos, with_labels=True)
# 标注边权
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)

# 获得最短路径
path = convert_to_tuples(shortest_path(G, 1, 6)[1])
# 标记最短路径
nx.draw_networkx_edges(G, pos, edgelist=path, edge_color='r')
# Display the graph
plt.show()
