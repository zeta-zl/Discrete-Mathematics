import heapq


def build_huffman_tree(freq):
    heap = [(freq[w], w) for w in freq]
    heapq.heapify(heap)

    while len(heap) > 1:
        f1, c1 = heapq.heappop(heap)
        f2, c2 = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, (c1, c2)))

    return heap[0][1]


def generate_huffman_code(tree, prefix=""):
    if isinstance(tree, tuple):
        generate_huffman_code(tree[0], prefix + "0")
        generate_huffman_code(tree[1], prefix + "1")
    else:
        huffman_code[tree] = prefix


# Example usage
freq = {'a': 6, 'b': 3, 'c': 8, 'd': 2, 'e': 10, 'f': 4}
tree = build_huffman_tree(freq)
huffman_code = {}
generate_huffman_code(tree)
print(huffman_code)
