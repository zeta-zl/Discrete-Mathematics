import matplotlib.pyplot as plt
import networkx as nx


def show_weighted_graph(G):
    """
    绘制带权图。
    """
    # 计算点的位置
    pos = nx.spring_layout(G, seed=0)

    # 绘制边
    nx.draw(G, pos, with_labels=True, node_size=300)

    # 添加边权标签
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

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


# 将给定的图G转换为多重图，并添加paths中的所有边
def add_paths(G, paths):
    """
    将给定的图G转换为多重图，并添加paths中的所有边

    参数：
    G -- 一个NetworkX图对象
    paths -- 一个路径列表，其中每条路径都是元组列表，表示路径中的边

    返回值：
    得到的多重图
    """
    MG = nx.MultiGraph()
    MG.add_nodes_from(G.nodes())
    MG.add_weighted_edges_from(G.edges(data='weight'))
    for path in paths:
        for edge in path:
            MG.add_edge(edge[0], edge[1], weight=G[edge[0]][edge[1]]['weight'])
    return MG


def chinese_postman(G, start=None):
    """
    解决中国邮政员问题，传入图G，返回邮政员的路径和总长
    参数：
    G -- 一个NetworkX图对象

    返回值：
    一个元组，包含欧拉回路路径和总长
    """
    # 奇点集
    odd_vertices = []
    # 确定图是否有欧拉回路
    if not nx.is_eulerian(G):
        # 标识奇点
        odd_vertices = [v for v in G.nodes() if G.degree(v) % 2 == 1]
    # 奇点之间的最短路径集合
    paths = []
    # 当图中存在奇点时
    while odd_vertices:
        # 计算奇点集中两个点之间的最短路径
        u, v, path = min_shortest_path(G, odd_vertices)
        # 添加最短路径到集合中
        paths.append(path)
        # 从奇点集中删除已经连接的点
        odd_vertices.remove(u)
        odd_vertices.remove(v)

    # 在原图中添加所有奇点之间的最短路径，得到多重图MG
    MG = add_paths(G, paths)

    # 在多重图MG中求出欧拉回路
    cp_path = list(nx.eulerian_path(MG, start))
    # 计算欧拉回路的总长
    cp_length = sum(MG[cp_path[i][0]][cp_path[i][1]][0]['weight'] for i in range(len(list(cp_path))))
    # 返回欧拉回路路径和总长
    return cp_path, cp_length


# 创建一个带权图
def graph_with_weight():
    """
    从键盘输入若干个三元组，并返回生成的图对象。
    """
    # 创建一个带权图
    G = nx.Graph()

    # 循环输入三元组
    print("输入边的起点、终点、边权，以空格分隔")
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


# 将元组列表转换为路径
def convert_to_string(tuples):
    # 创建一个空字符串来存储结果
    result = ""

    # 循环遍历元组
    for t in tuples:
        # 将元组的第一个元素添加到结果字符串中
        result += t[0]

    # 将起点添加到末尾
    result += result[0]

    # 返回结果
    return result


if __name__ == "__main__":
    G = graph_with_weight()
    p, l = chinese_postman(G, 'a')
    print(convert_to_string(p), l, sep='\n')
    show_weighted_graph(G)

"""
测试用例
a b 7
b c 6
c d 6
a e 6
e f 4
b f 5
f g 5
c g 5
g h 5
d h 6
e i 3
i j 2
f j 2
j k 4
g k 2
k h 4
k l 3
h l 3
"""
