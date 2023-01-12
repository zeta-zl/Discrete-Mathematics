def edge_list_to_adj_matrix(edge_list):
    edge_list = [tuple(map(int, edge.split())) for edge in edge_list.split("\n") if edge]
    vertex_list = set([vertex for edge in edge_list for vertex in edge])
    n = max(vertex_list)
    adj_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for edge in edge_list:
        adj_matrix[edge[0] - 1][edge[1] - 1] = 1
    return adj_matrix


# 示例
edge_list = "1 2\n2 3\n3 4\n4 5"
adj_matrix = edge_list_to_adj_matrix(edge_list)
print(adj_matrix)
