import networkx as nx
import matplotlib.pyplot as plt


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


def fleury(G, source):
    path = []
    current_node = source
    while G.number_of_edges() > 0:
        # 获得当前点的邻居
        neighbors = list(G.neighbors(current_node))
        if len(neighbors) > 0:
            if len(neighbors) == 1:
                # 如果只有一个邻居
                next_node = neighbors[0]
            else:
                # 如果多于一个邻居
                if is_bridge(G, current_node, neighbors[0]):
                    # 如果与第一个邻居之间的边是桥，就走第二个邻居
                    # 注意一个点不可能同时与两座桥相连，否则不存在欧拉回路
                    next_node = neighbors[1]
                else:
                    next_node = neighbors[0]

            G.remove_edge(current_node, next_node)
            path.append(next_node)
            current_node = next_node
        else:
            # 如果没有邻居，代表走入死路。此时回溯
            path.pop()
            current_node = path[-1]
    return path


def is_bridge(G, u, v):
    # create a copy of the graph to perform the test on
    H = G.copy()

    # remove the edge between the two nodes
    H.remove_edge(u, v)

    # check if the graph is still connected
    return nx.is_connected(H)


# create a graph with 5 nodes and 7 edges
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (2, 0), (0, 3), (3, 1), (1, 4), (4, 2)])
G1 = G.copy()

# find the Eulerian path starting at node 0
path = fleury(G, 0)
print(path)
plot_graph(G1)
