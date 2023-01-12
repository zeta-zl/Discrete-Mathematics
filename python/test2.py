import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# 绘制图
def plot_graph(G, style=False, pos=None):
    """
      绘制给定的图像。

      参数：
      G -- 一个networkx中的图对象
      style -- 可选参数，如果设置为True，则使用不同的样式染色
      pos -- 可选参数，表示节点位置的字典
      """
    # 使用pos参数绘制点
    if pos is None:
        pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, with_labels=True)

    # 如果需要，染色
    if style:
        styles = {0: "o", 1: '^', 2: 'd', 3: 'v'}  # 点的样式,'so^>v<dph8'中的一个
        node_styles = nx.greedy_color(G)
        for n in G.nodes():
            nx.draw_networkx_nodes(n, pos, node_shape=styles[node_styles[n]])

    plt.show()


G = nx.Graph()
G.add_edge('A', 'B')
G.add_edge('B', 'C')

plot_graph(G, style=True)
