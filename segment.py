import networkx as nx
import numpy as np
from typing import List, Tuple

DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0), (-1, -1), (1, 1), (-1, 1), (1, -1)]


def segment_image(k: int, image: np.ndarray) -> np.ndarray:
    """
    Segment a grayscale image into n segments with MST-based segmentation.
    Create an undirected graph representation of the image and
    run the described modified Kruskal's algorithm to form segments

    :param k: the parameter to determine if a segment should be merged
    :param image: a 2D np.ndarray representing a grayscale image
    :return: a 2D np.ndarray the same shape as the original image where each entry
    is a label corresponding to the segment the pixel belongs to
    """

    assert len(image.shape) == 2

    # get the length of the image
    n = len(image)
    m = len(image[0])

    vertices = []
    values = []

    # get vertices and values to build UnionFind
    for i in range(n):
        for j in range(m):
            vertices.append((i, j))
            values.append(image[i][j])

    uf = UnionFind(vertices, values)

    # get all edges and sort by weight
    edges = get_edges(image, n, m)
    edges.sort()

    # the core of kruskal's algorithm
    for weight, u, v in edges:
        u_root = uf.find(u)
        v_root = uf.find(v)
        threshold = min(uf.max_diff(u) + k / uf.size(u), uf.max_diff(v) + k / uf.size(v))
        # if the edge does not create a cycle and the weight is under the threshold, union two segments
        if u_root != v_root and weight <= threshold:
            uf.union(u, v)

    # initiate the result ndarray
    res = np.array([[-1] * m] * n)

    # get all roots (the ultimate parent of each segment)
    roots = get_roots(uf)

    # assign a unique label to each segment
    root_to_label = {}
    label = 0
    for root in roots:
        root_to_label[root] = label
        label += 1

    # for each pixel, label all parents till the root is reached
    for i in range(n):
        for j in range(m):
            # if the pixel is already labeled, ignore
            if res[i][j] != -1:
                continue
            # get the root of the current pixel
            curr_root = uf.find((i, j))
            # get the corresponding label for all the pixels under the current pixel's root,
            # i.e. in the same segment with the root
            curr_label = root_to_label[curr_root]
            # label all parents till the root is reached
            label_connected_component(i, j, uf, curr_label, res)

    return res


def get_roots(uf: 'UnionFind') -> List[Tuple[int, int]]:
    """get the pixels whose parent is themselves, i.e. root of the segment

    :param uf: a union find data structure
    :return: a list of pixels (horizontal index and vertical index) whose parent is themselves
    """
    return [x for x in uf.parent if uf.parent[x] == x]


def label_connected_component(i: int, j: int, uf: 'UnionFind', label: int, res: np.ndarray) -> None:
    """label all pixels within the same connected component, i.e. segment

    :param i: the horizontal index of the current pixel
    :param j: the vertical index of the current pixel
    :param uf: the union find data structure
    :param label: the label (an integer) assigned for this segment
    :param res: a 2D np.ndarray the same shape as the original image where each entry
    is a label corresponding to the segment the pixel belongs to
    """
    curr = (i, j)
    while uf.parent[curr] != curr:
        res[curr[0]][curr[1]] = label
        curr = uf.parent[curr]
    res[curr[0]][curr[1]] = label


def get_edges(image: np.ndarray, n: int, m: int) -> List:
    """get all edges from the image

    :param image: a 2D np.ndarray representing a grayscale image
    :param n: the number of rows in image
    :param m: the number of columns in image
    :return: a list of all edges in the format [[weight, pos1(i, j), pos2(x, y)], [weight, pos2(a, b), pos3(p, q)], ...]
    """
    edges = []
    for i in range(n):
        for j in range(m):
            # iterate through possible neighbors in the eight directions
            for dx, dy in DIRECTIONS:
                next_x = i + dx
                next_y = j + dy
                # if the index of a neighbor is beyond the boundary, ignore
                if not is_valid_position(next_x, next_y, n, m):
                    continue
                weight = abs(image[i][j] - image[next_x][next_y])
                edges.append([weight, (i, j), (next_x, next_y)])
    return edges


def is_valid_position(next_x: int, next_y: int, n: int, m: int) -> bool:
    """check if the index of a neighbor is valid

    :param next_x: the horizontal index of the neighbor
    :param next_y: the vertical index of the neighbor
    :param n: the number of rows in image
    :param m: the number of columns in image
    :return: true if the index is valid, false otherwise
    """
    return 0 <= next_x < n and 0 <= next_y < m


class UnionFind:
    def __init__(self, vertices, values):
        """
        :param vertices: a list of vertices, must be hashable
        :param values: a list of corresponding grayscale values for the vertices
        """
        # the parent of x, used by find() to retrieve x's segment
        self.parent = {x: x for x in vertices}
        # a map from x to the rank of x's segment
        self.rank = {x: 0 for x in vertices}
        # the number of vertices in x's segment
        self.sizes = {x: 1 for x in vertices}
        # the smallest value in x's segment
        self.min_values = {x: values[idx] for idx, x in enumerate(vertices)}
        # the largest value in x's segment
        self.max_values = {x: values[idx] for idx, x in enumerate(vertices)}

    def find(self, x):
        """
        :param x: a vertex
        :return: the segment that x belongs to
        """
        if self.parent[x] == x:
            return x
        return self.find(self.parent[x])

    def union(self, x, y):
        """
        combines the two segments connected by the edge (x,y) into one segment
        :param x: a vertex
        :param y: a vertex
        """
        x_root = self.find(x)
        y_root = self.find(y)
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
            self.sizes[y_root] += self.sizes[x_root]
            self.min_values[y_root] = min(self.min_values[y_root], self.min_values[x_root])
            self.max_values[y_root] = max(self.max_values[y_root], self.max_values[x_root])
        else:
            self.parent[y_root] = x_root
            self.sizes[x_root] += self.sizes[y_root]
            self.min_values[x_root] = min(self.min_values[x_root], self.min_values[y_root])
            self.max_values[x_root] = max(self.max_values[x_root], self.max_values[y_root])
            if self.rank[x_root] == self.rank[y_root]:
                self.rank[x_root] += 1

    def size(self, x):
        """
        :param x: a vertex
        :return: returns the number of vertices in x's segment
        """
        return self.sizes[self.find(x)]

    def max_diff(self, x):
        """
        :param x: a vertex
        :return: the largest difference in values between any vertices in x's segment
        """
        return self.max_values[self.find(x)] - self.min_values[self.find(x)]


if __name__ == "__main__":
    pass
    # import utils
    #
    # utils.load_segment_save_image(9, "wave.png", "wave_segmented.png")
    # utils.load_segment_save_image(3, "smiling.png", "smiling_segmented.png")
