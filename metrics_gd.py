import numpy as np
import copy
import networkx as nx
import time

from shapely.geometry import LineString


def crossing_res_and_crossings_metric(coords, gtds):

    start = time.time()
    cnt = 0
    curr_min_angle = 360

    adj_matrix = copy.deepcopy(gtds)
    adj_matrix[adj_matrix > 1] = 0
    graph = nx.from_numpy_array(adj_matrix)
    graph = nx.convert_node_labels_to_integers(graph)

    edges = list(graph.edges())

    # loop over all the edges
    for i in range(len(edges)):
        edge1 = edges[i]

        # get the line of the first edge
        c1 = [(coords[edge1[0]][0], coords[edge1[0]][1]), (coords[edge1[1]][0], coords[edge1[1]][1])]

        # it can happen that the layout algorithm puts two nodes on the EXACT same position, we simply continue with the next edge
        if c1[0] == c1[1]:
            continue

        slope1 = (c1[1][1] - c1[0][1]) / (c1[1][0] - c1[0][0])
        first_line = LineString(c1)

        # loop over the other edges starting from i (duplicate crossings won't be counted then)
        for j in range(i, len(edges)):
            edge2 = edges[j]

            # only check if edges cross if they do not share a node
            if edge1[0] not in edge2 and edge1[1] not in edge2:
                c2 = [(coords[edge2[0]][0], coords[edge2[0]][1]), (coords[edge2[1]][0], coords[edge2[1]][1])]
                second_line = LineString(c2)

                # if there is an intersection increase the ocunt
                if first_line.intersects(second_line):
                    slope2 = (c2[1][1] - c2[0][1]) / (c2[1][0] - c2[0][0])
                    angle = abs((slope2 - slope1) / (1 + slope1 * slope2))
                    deg_angle = np.arctan(angle) * 180 / np.pi
                    cnt += 1

                    if deg_angle < curr_min_angle:
                        curr_min_angle = deg_angle

    if curr_min_angle == 360:
        cross_res = 1
    else:
        cross_res = curr_min_angle / 90

    # this calculation is used if you're looping over all edges twice, in our case we skip duplicate comparisons
    #cnt = cnt / (len(edges)**2)

    cnt = 1 - (cnt / (len(edges) * (len(edges) - 1) / 2))

    #print('cr and cn ' + str(round(time.time() - start, 2)))
    return cross_res, cnt


"""
A simple function for computing the overall stress of the current layout

Input
final_g_dict:   dict, a dictionary containg the coordinates "coords", a graph object G "G", the graph theoretic distances "gtds" and the stress alpha "stress_alpha"

Output
stress_tot:     float, the current total number of stress
"""


def norm_stress_node_resolution_metric(coords, gtds, stress_alpha):

    start = time.time()

    with np.errstate(divide = 'ignore'):
        # compute the weights and set the numbers that turned to infinity (the 0 on the diagonals) to 0
        weights = np.array(gtds).astype(float) ** -stress_alpha
        weights[weights == float('inf')] = 0

    # calculate the euclidean distances
    eucl_dis = np.sqrt(np.sum(((np.expand_dims(coords, axis = 1) - coords) ** 2), 2))

    # node resolution computation
    max_dis = np.max(eucl_dis[np.nonzero(eucl_dis)])
    min_dis = np.min(eucl_dis[np.nonzero(eucl_dis)])
    nr = min_dis / max_dis

    # compute stress by multiplying the distances with the graph theoretic distances, squaring it, multiplying by the weight factor and then taking the sum
    stress_tot = np.sum(weights * ((eucl_dis - gtds) ** 2))
    ns = 1 - (stress_tot / (np.shape(coords)[0] ** 2))

    #print('ns' + str(round(time.time() - start, 2)))

    #return ns, nr
    return ns


"""
A simple function for computing the angular resolution of the current layout

Input
final_g_dict:   dict, a dictionary containg the coordinates "coords", a graph object G "G", the graph theoretic distances "gtds" and the stress alpha "stress_alpha"

Output
res:            float, the angular resolution
"""


def angular_resolution_metric(coords, gtds):

    adj_matrix = copy.deepcopy(gtds)
    adj_matrix[adj_matrix > 1] = 0
    graph = nx.from_numpy_array(adj_matrix)
    graph = nx.convert_node_labels_to_integers(graph)

    # initialize variables
    n = graph.number_of_nodes()
    nodes = list(graph.nodes())
    angle_res = 360

    # get the maximum degree
    max_degr = max(list(dict(graph.degree).values()))

    # loop over all nodes
    for i in range(n):
        # only compute angles if there are at least 2 edges to a node
        if graph.degree(nodes[i]) > 1:
            curr_neighbs = list(graph.neighbors(nodes[i]))

            # get the ordering and then get the angles of that specific ordering
            order_neighbs = compute_order(curr_node = nodes[i], neighbors = curr_neighbs, coords = coords)
            norm_sub = np.subtract(coords[order_neighbs, ].copy(), coords[nodes[i], ])
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

                if angle < angle_res:
                    angle_res = angle

                sub_phi = np.delete(sub_phi, 0)

    res = np.radians(angle_res) / (2 * np.pi / max_degr)

    if not isinstance(res, np.float64):
        res = res[0]

    #print('ar' + str(round(time.time() - start, 2)))

    return res


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