def muti(a, b, c=2):
    return (a + b) % c


def find_subgroups(G):
    # 求出群 G 的所有子集
    all_subsets = []
    for i in range(1, 2 ** len(G)):
        subset = []
        for j in range(len(G)):
            if i & (1 << j):
                subset.append(G[j])
        all_subsets.append(subset)

    # 检查所有子集是否是子群
    subgroups = []
    for subset in all_subsets:
        is_subgroup = True
        for a in subset:
            for b in subset:
                if not muti(a, b, len(G)) in subset:
                    is_subgroup = False
                    break
            if not is_subgroup:
                break
        if is_subgroup:
            subgroups.append(subset)

    return subgroups


if __name__ == "__main__":
    g = list(map(eval, input().split()))
    print(g)
    print(find_subgroups(g))
