import matplotlib.pyplot as plt
import networkx as nx
import itertools


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


def is_subgraph(G1, G2):
    # 检查G2是否是G1的子图
    return nx.algorithms.isomorphism.GraphMatcher(G1, G2).subgraph_is_isomorphic()


def all_subgraphs(G):
    """
    生成给定图G的所有子图的列表。

    参数：
    G -- NetworkX图对象

    返回值：
    G的子图的列表，其中每个子图与G具有相同的节点集，但边集是G的子集。
    """
    # 获取G的所有边的子集
    edge_subsets = itertools.chain.from_iterable(
        itertools.combinations(G.edges(), r) for r in range(len(G.edges()) + 1))
    # 创建子图列表
    subgraphs = []
    for edges in edge_subsets:
        # 从边创建子图
        subgraph = nx.Graph()
        subgraph.add_edges_from(edges)
        # 添加来自G的节点
        subgraph.add_nodes_from(G.nodes())
        # 将子图添加到列表中
        subgraphs.append(subgraph)

    return subgraphs


# 不需要输入
if __name__ == "__main__":
    G_6 = nx.complete_graph(6)
    G_3 = nx.complete_graph(3)
    subgraphs = all_subgraphs(G_6)

    for i in subgraphs:
        # 如果子图i和i的补图都不包含3阶完全图
        if not (is_subgraph(i, G_3) or is_subgraph(nx.complement(i), G_3)):
            print('wrong')
            break
    else:
        print('r(3,3)=6')
