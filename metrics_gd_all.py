import numpy as np
import copy
import networkx as nx
import shapely as sh


def all_metrics(coords, gtds, stress_alpha = 2, diameter = 1/60, edge_overlap_margin = 0.01):

    curr_min_angle = 360
    cross_count = 0
    node_node_cnt = 0
    max_nn_cnt = 0
    node_edge_cnt = 0
    max_ne_cnt = 0
    edge_edge_cnt = 0
    max_ee_cnt = 0

    node_polys = {}

    adj_matrix = copy.deepcopy(gtds)
    adj_matrix[adj_matrix > 1] = 0
    graph = nx.from_numpy_array(adj_matrix)
    graph = nx.convert_node_labels_to_integers(graph)

    edges = list(graph.edges())
    node_pairs_compared = {}
    node_edge_pairs_compared = {}

    angle_res = 360

    # get the maximum degree
    max_degr = max(list(dict(graph.degree).values()))

    # loop over all the edges
    for i in range(len(edges)):
        edge1 = edges[i]
        node1 = edge1[0]
        node2 = edge1[1]

        # angular resolution
        for n in [node1, node2]:
            # only compute angles if there are at least 2 edges to a node
            if graph.degree(n) > 1:
                curr_neighbs = list(graph.neighbors(n))

                # get the ordering and then get the angles of that specific ordering
                order_neighbs = compute_order(curr_node=n, neighbors=curr_neighbs, coords=coords)
                norm_sub = np.subtract(coords[order_neighbs, ].copy(), coords[n, ])
                sub_phi = np.arctan2(norm_sub[:, 1:2], norm_sub[:, :1]) * 180 / np.pi

                # now compare each consecutive edge pair to get the smallest seen angle
                while len(sub_phi) >= 2:
                    first = sub_phi[0]
                    second = sub_phi[1]

                    # can simply subtract the angles in these cases
                    if (first >= 0 and second >= 0) or (first <= 0 and second <= 0) or (first >= 0 and second <= 0):
                        angle = abs(first - second)
                    # have to add 360 for this case
                    elif (first < 0 and second > 0):
                        angle = 360 + first - second

                    if angle <= angle_res:
                        angle_res = angle

                    sub_phi = np.delete(sub_phi, 0)


        # for metrics that compare nodes or loop over nodes, we do this within this loop that loops over edges
        # compare nodes of first edge
        if (tuple([node1, node2]) not in node_pairs_compared) and (tuple([node2, node1]) not in node_pairs_compared):
            node_pairs_compared[tuple([node1, node2])] = 1

            # node node occlusion
            node_polys[node1] = sh.Point(coords[node1]).buffer(diameter)
            node_polys[node2] = sh.Point(coords[node2]).buffer(diameter)
            max_nn_cnt += 1
            if node_polys[node1].intersects(node_polys[node2]):
                node_node_cnt += 1

        # get the line of the first edge
        c1 = [(coords[edge1[0]][0], coords[edge1[0]][1]), (coords[edge1[1]][0], coords[edge1[1]][1])]
        first_line = sh.LineString(c1)

        # make sure the nodes of the first edge aren't on the exact same positions
        if c1[0] != c1[1]:
            slope1 = (c1[1][1] - c1[0][1]) / (c1[1][0] - c1[0][0])


        # loop over the other edges starting from i (duplicate crossings won't be counted then)
        for j in range(i, len(edges)):
            edge2 = edges[j]
            node3 = edge2[0]
            node4 = edge2[1]

            node_polys[node3] = sh.Point(coords[node3]).buffer(diameter)
            node_polys[node4] = sh.Point(coords[node4]).buffer(diameter)

            # node node intersections
            # compare nodes of 2nd edge with each other
            if (tuple([node3, node4]) not in node_pairs_compared) and (tuple([node3, node4]) not in node_pairs_compared):
                node_pairs_compared[tuple([node3, node4])] = 1
                max_nn_cnt += 1
                if node_polys[node3].intersects(node_polys[node4]):
                    node_node_cnt += 1

            # compare nodes of edges
            if (tuple([node1, node3]) not in node_pairs_compared) and (tuple([node3, node1]) not in node_pairs_compared):
                node_pairs_compared[tuple([node1, node3])] = 1
                max_nn_cnt += 1
                if node_polys[node1].intersects(node_polys[node3]):
                    node_node_cnt += 1

            if (tuple([node2, node4]) not in node_pairs_compared) and (tuple([node2, node4]) not in node_pairs_compared):
                node_pairs_compared[tuple([node2, node4])] = 1
                max_nn_cnt += 1
                if node_polys[node2].intersects(node_polys[node4]):
                    node_node_cnt += 1

            if (tuple([node2, node3]) not in node_pairs_compared) and (tuple([node2, node3]) not in node_pairs_compared):
                node_pairs_compared[tuple([node2, node3])] = 1
                max_nn_cnt += 1
                if node_polys[node2].intersects(node_polys[node3]):
                    node_node_cnt += 1



            c2 = [(coords[edge2[0]][0], coords[edge2[0]][1]), (coords[edge2[1]][0], coords[edge2[1]][1])]
            second_line = sh.LineString(c2)

            # check the nodes of the second edge and check if they are not in the first edge
            if (node3 not in edge1) and (node4 not in edge1):
                # check if the first node of the second edge intersects with the first edge
                if (tuple([node3, node1, node2]) not in node_edge_pairs_compared) and (tuple([node3, node2, node1]) not in node_edge_pairs_compared):
                    node_edge_pairs_compared[tuple([node3, node1, node2])] = 1
                    max_ne_cnt += 1
                    if node_polys[node3].intersects(second_line):
                        node_edge_cnt += 1
                # check if the second node of the second edge intersects with the first edge
                if (tuple([node4, node1, node2]) not in node_edge_pairs_compared) and (tuple([node4, node2, node1]) not in node_edge_pairs_compared):
                    node_edge_pairs_compared[tuple([node4, node1, node2])] = 1
                    max_ne_cnt += 1
                    if node_polys[node4].intersects(second_line):
                        node_edge_cnt += 1

            # make sure the first node and second node arent in the second edge
            if node1 not in edge2:
                # node edge intersections
                # check if the first node of the first edge intersect with the second edge
                if (tuple([node1, node3, node4]) not in node_edge_pairs_compared) and (tuple([node1, node4, node3]) not in node_edge_pairs_compared):
                    node_edge_pairs_compared[tuple([node1, node3, node4])] = 1
                    max_ne_cnt += 1
                    if node_polys[node1].intersects(second_line):
                        node_edge_cnt += 1

                # make sure the nodes of the second edge aren't on the exact same positions
                if node2 not in edge2:
                    # node edge intersections
                    # check if the second node of the first edge intersect with the second edge
                    if (tuple([node2, node3, node4]) not in node_edge_pairs_compared) and (tuple([node2, node4, node3]) not in node_edge_pairs_compared):
                        node_edge_pairs_compared[tuple([node2, node3, node4])] = 1
                        max_ne_cnt += 1
                        if node_polys[node2].intersects(second_line):
                            node_edge_cnt += 1

                    if c2[0] != c2[1]:
                        max_ee_cnt += 1
                        if first_line.intersects(second_line):
                            slope2 = (c2[1][1] - c2[0][1]) / (c2[1][0] - c2[0][0])
                            angle = abs((slope2 - slope1) / (1 + slope1 * slope2))
                            deg_angle = np.arctan(angle) * 180 / np.pi
                            cross_count += 1

                            if deg_angle < curr_min_angle:
                                curr_min_angle = deg_angle


                            if (slope2 - slope1) < edge_overlap_margin:
                                edge_edge_cnt += 1





    if curr_min_angle == 360:
        cr = 1
    else:
        cr = curr_min_angle / 90

    cn = 1 - (cross_count / (len(edges) * (len(edges) - 1) / 2))

    nn = 1 - (node_node_cnt / max_nn_cnt)

    ne = 1 - (node_edge_cnt / max_ne_cnt)

    ee = 1 - (edge_edge_cnt / max_ee_cnt)

    ar = np.radians(angle_res) / (2 * np.pi / max_degr)

    if not isinstance(ar, np.float64):
        ar = ar[0]

    with np.errstate(divide = 'ignore'):
        # compute the weights and set the numbers that turned to infinity (the 0 on the diagonals) to 0
        weights = np.array(gtds).astype(float) ** -stress_alpha
        weights[weights == float('inf')] = 0

    # calculate the euclidean distances
    eucl_dis = np.sqrt(np.sum(((np.expand_dims(coords, axis = 1) - coords) ** 2), 2))

    # compute stress by multiplying the distances with the graph theoretic distances, squaring it, multiplying by the weight factor and then taking the sum
    stress_tot = np.sum(weights * ((eucl_dis - gtds) ** 2))
    ns = 1 - (stress_tot / (np.shape(coords)[0] ** 2))

    return ns, cr, ar, nn, ne, cn, ee




"""
Function that computes the order of neighbors around a node in clockwise-order starting at 12 o'clock
Input
curr_node:      int, the integer id of the node for which we want to know the order
neighbors:      list, a list of integer ids of the neighbors of the current node
coords:         np.array or tensor, a 2xn array or tensor of x,y node coordinates

Output
neighbors:      list, the ordered list of neighbors
"""


def compute_order(curr_node, neighbors, coords):
    # get the center x and y coordinate
    center_x = coords[curr_node][0]
    center_y = coords[curr_node][1]

    # loop over all the neighbors except the last one
    for i in range(len(neighbors) - 1):
        curr_min_idx = i

        # loop over the other neighbors
        for j in range(i + 1, len(neighbors)):

            a = coords[neighbors[j]]
            b = coords[neighbors[curr_min_idx]]

            # compare the points to see which node comes first in the ordering
            if compare_points(a[0], a[1], b[0], b[1], center_x, center_y):
                curr_min_idx = j

        if curr_min_idx != i:
            neighbors[i], neighbors[curr_min_idx] = neighbors[curr_min_idx], neighbors[i]

    return neighbors


"""
Function that compares two points (nodes) to each other to determine which one comes first w.r.t. a center
Original solution from https://stackoverflow.com/questions/6989100/sort-points-in-clockwise-order

Input
a_x:            float, the x coordinate of the first node
a_y:            float, the y coordinate of the first node
b_x:            float, the x coordinate of the second node
b_y:            float, the y coordinate of the second node
center_x:       float, the x coordinate of the center node (curr_node from compute_order function)
center_y:       float, the y coordinate of the center node (curr_node from compute_order function)

Output
res:            boolean, if True then a comes before b
"""


def compare_points(a_x, a_y, b_x, b_y, center_x, center_y):
    if ((a_x - center_x) >= 0 and (b_x - center_x) < 0):
        return True

    if ((a_x - center_x) < 0 and (b_x - center_x) >= 0):
        return False

    if ((a_x - center_x) == 0 and (b_x - center_x) == 0):
        if ((a_y - center_y) >= 0 or (b_y - center_y) >= 0):
            return a_y > b_y
        return b_y > a_y

    # compute the cross product of vectors (center -> a) x (center -> b)
    det = (a_x - center_x) * (b_y - center_y) - (b_x - center_x) * (a_y - center_y)
    if (det < 0):
        return True
    if (det > 0):
        return False

    # points a and b are on the same line from the center
    # check which point is closer to the center
    d1 = (a_x - center_x) * (a_x - center_x) + (a_y - center_y) * (a_y - center_y)
    d2 = (b_x - center_x) * (b_x - center_x) + (b_y - center_y) * (b_y - center_y)

    res = d1 > d2

    return res