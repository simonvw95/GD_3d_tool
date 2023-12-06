import numpy as np
import copy
import networkx as nx
import shapely as sh


"""
Crossing resolution deviation of the current layout

Input
coords:         np.ndarray, a 2xn numpy array containing the x,y coordinates
gtds:           np.ndarray, an nxn numpy array containing the graph theoretic distances (shortest path lengths)

Output
cr:             float, the crossing resolution deviation
"""


def crossing_res_dev(coords, gtds):

    # construct a graph from the shortest path matrix
    adj_matrix = copy.deepcopy(gtds)
    adj_matrix[adj_matrix > 1] = 0
    graph = nx.from_numpy_array(adj_matrix)
    graph = nx.convert_node_labels_to_integers(graph)

    edges = list(graph.edges())
    m = len(edges)

    polys = [0] * m

    # loop over all the edges
    for i in range(m):
        edge1 = edges[i]

        # get the line of the first edge
        c1 = [(coords[edge1[0]][0], coords[edge1[0]][1]), (coords[edge1[1]][0], coords[edge1[1]][1])]
        polys[i] = sh.LineString(c1)

    # construct an STR-tree containing the edge objects
    s = sh.STRtree(polys)

    cross_angles = []

    # loop over all the edges
    for i in polys:
        i_coords = i.xy

        # check which edges cross the current one
        res = s.query(i, predicate='crosses')
        for j in res:
            j_coords = polys[j].xy

            x_diff_1 = (i_coords[1][0] - i_coords[0][0])
            y_diff_1 = (i_coords[1][1] - i_coords[0][1])
            x_diff_2 = (j_coords[1][1] - j_coords[0][1])
            y_diff_2 = (j_coords[1][0] - j_coords[0][0])

            # if we have a vertical line then slightly shift both lines
            if (x_diff_1 == 0) or (x_diff_2 == 0):
                x_diff_1 = 0.01
                x_diff_2 = 0.01

            # get the angle of the intersection
            slope1 = y_diff_1 / x_diff_1
            slope2 = y_diff_2 / x_diff_2
            angle = abs((slope2 - slope1) / (1 + slope1 * slope2))
            deg_angle = np.arctan(angle) * 180 / np.pi

            cross_angles.append(deg_angle)

    # if there are no crossings, crossing resolution is 1
    if cross_angles:
        cr = np.mean(np.array(cross_angles) / 90)
    else:
        cr = 1

    return cr


"""
Crossing number of the current layout

Input
coords:         np.ndarray, a 2xn numpy array containing the x,y coordinates
gtds:           np.ndarray, an nxn numpy array containing the graph theoretic distances (shortest path lengths)

Output
cn:             float, the crossing number
"""


def crossings_number(coords, gtds):

    # construct a graph from the shortest path matrix
    adj_matrix = copy.deepcopy(gtds)
    adj_matrix[adj_matrix > 1] = 0
    graph = nx.from_numpy_array(adj_matrix)
    graph = nx.convert_node_labels_to_integers(graph)

    edges = list(graph.edges())
    m = len(edges)
    polys = [0] * m

    cnt = 0

    # loop over all the edges
    for i in range(m):
        edge1 = edges[i]
        # get a line object
        c1 = [(coords[edge1[0]][0], coords[edge1[0]][1]), (coords[edge1[1]][0], coords[edge1[1]][1])]
        polys[i] = sh.LineString(c1)

    s = sh.STRtree(polys)

    # STRtree query with predicate crosses does not count incident edges nor the same edge object (2 identical edge with identical edge lengths) as a crossing
    # so maximum number of crossings is not m^2 but:
    # (m * (m - 1) / 2) the maximum number of total crossings (not including duplicate comparisons)
    # minus 1/2 * sum(degrees * (degrees - 1))
    for i in polys:
        res = s.query(i, predicate = 'crosses')
        cnt += len(res)

    end_cnt = cnt / 2  # duplicate comparisons done, divide by half to get the actual crossings
    cr_poss = m * (m - 1) / 2
    degrees = np.array(list(dict(graph.degree()).values()))
    cr_imp = np.sum(degrees * (degrees - 1)) / 2

    cn = 1 - (end_cnt / (cr_poss - cr_imp))

    return cn


"""
Normalized stress of the current layout

Input
coords:         np.ndarray, a 2xn numpy array containing the x,y coordinates
gtds:           np.ndarray, an nxn numpy array containing the graph theoretic distances (shortest path lengths)
stress_alpha:   int, a value used to weight the gtds, default set to 2

Output
ns:             float, the normalized stress
"""


def norm_stress(coords, gtds, stress_alpha = 2):

    with np.errstate(divide = 'ignore'):
        # compute the weights and set the numbers that turned to infinity (the 0 on the diagonals) to 0
        weights = np.array(gtds).astype(float) ** -stress_alpha
        weights[weights == float('inf')] = 0

    # calculate the euclidean distances
    eucl_dis = np.sqrt(np.sum(((np.expand_dims(coords, axis = 1) - coords) ** 2), 2))
    # scale the coordinates
    scal_coords = coords * (np.nansum((eucl_dis / gtds) / np.nansum((eucl_dis ** 2) / (gtds ** 2))))

    # compute the euclidean distances again according to scaled coordinates
    eucl_dis_new = np.sqrt(np.sum(((np.expand_dims(scal_coords, axis=1) - scal_coords) ** 2), 2))

    # compute stress
    stress_tot = np.sum(weights * ((eucl_dis_new - gtds) ** 2))
    ns = 1 - (stress_tot / (np.shape(coords)[0] ** 2))

    return ns


"""
Edge length deviation the current layout

Input
coords:     np.ndarray, a 2xn numpy array containing the x,y coordinates
gtds:       np.ndarray, an nxn numpy array containing the graph theoretic distances (shortest path lengths)

Output
el:         float, the edge length deviation
"""


def edge_lengths_sd(coords, gtds):

    # construct a graph from the shortest path matrix
    adj_matrix = copy.deepcopy(gtds)
    adj_matrix[adj_matrix > 1] = 0
    graph = nx.from_numpy_array(adj_matrix)
    graph = nx.convert_node_labels_to_integers(graph)

    edges = list(graph.edges())
    m = len(edges)

    # get the euclidean distances
    eucl_dis = np.sqrt(np.sum(((np.expand_dims(coords, axis = 1) - coords) ** 2), 2))

    # now only get the euclidean distances of the edges
    edge_dis = eucl_dis[np.array(edges)[:, 0], np.array(edges)[:, 1]]

    mu = np.mean(edge_dis)

    # best edge length standard deviation is 0
    # so turn this around so that the best el value is 1
    el = 1 - (np.sqrt(np.sum((edge_dis - mu)**2) / m))

    return el


"""
Node resolution of the current layout

Input
coords:     np.ndarray, a 2xn numpy array containing the x,y coordinates

Output
nr:         float, the node resolution
"""


def node_resolution(coords):

    # calculate the euclidean distances
    eucl_dis = np.sqrt(np.sum(((np.expand_dims(coords, axis = 1) - coords) ** 2), 2))

    # exclude zeros
    eucl_dis = eucl_dis[np.nonzero(eucl_dis)]

    # ratio between smallest and largest eucl distance
    nr = np.min(eucl_dis) / np.max(eucl_dis)

    return nr


"""
Node-node occlusion of the current layout

Input
coords:     np.ndarray, a 2xn numpy array containing the x,y coordinates
r:          float, a value indicating the radius of a drawn node

Output
nn:         float, the node-node occlusion
"""


def node_node_occl(coords, r = 0.01):

    # get the number of nodes
    n = np.shape(coords)[0]

    node_polys = [0] * n
    for i in range(n):
        node_polys[i] = sh.Point(coords[i]).buffer(r)

    s = sh.STRtree(node_polys)

    areas = 0
    node_area = np.pi * (r**2)

    # compare each node with each other node
    for i in range(len(node_polys)):
        res = s.query(node_polys[i], predicate = 'intersects')

        for j in res:
            # exclude self intersects
            if i != j:
                curr_area = node_polys[i].intersection(node_polys[j]).area / node_area
                areas += curr_area

    nn = 1 - (areas / (n**2 - n))

    return nn


"""
Node-edge occlusion of the current layout

Input
coords:     np.ndarray, a 2xn numpy array containing the x,y coordinates
gtds:       np.ndarray, an nxn numpy array containing the graph theoretic distances (shortest path lengths)
r:          float, a value indicating the radius of a drawn node
width:      float, a value indicating the width of a drawn edge

Output
ne:         float, the node-edge occlusion
"""


def node_edge_occl(coords, gtds, r, width):

    # construct a graph from the shortest path matrix
    adj_matrix = copy.deepcopy(gtds)
    adj_matrix[adj_matrix > 1] = 0
    graph = nx.from_numpy_array(adj_matrix)
    graph = nx.convert_node_labels_to_integers(graph)

    edges = list(graph.edges())
    m = len(edges)
    nodes = list(graph.nodes())
    n = len(nodes)

    edges = np.array(edges)
    # get the y and x coordinates of the edges in their separate arrays
    y2s = coords[edges[:, 1]][:, 1]
    y1s = coords[edges[:, 0]][:, 1]
    x2s = coords[edges[:, 1]][:, 0]
    x1s = coords[edges[:, 0]][:, 0]

    # get the angles
    angles = np.arctan2(y2s - y1s, x2s - x1s)

    half_width = width / 2

    # calculate the offset for the corners
    offsets = half_width * np.array([-np.sin(angles), np.cos(angles)])

    r_off = r * np.array([np.cos(angles), np.sin(angles)])

    # find the corner points
    top_lefts = (np.array([x1s, y1s]) - offsets + r_off).T
    top_rights = (np.array([x2s, y2s]) - offsets - r_off).T
    bottom_lefts = (np.array([x1s, y1s]) + offsets + r_off).T
    bottom_rights = (np.array([x2s, y2s]) + offsets - r_off).T

    # store the node circles and edge rectangles
    node_polys = [0] * n
    edge_polys = [0] * m

    for i in range(n):
        node_polys[i] = sh.Point(coords[i]).buffer(r)

    for i in range(m):
        edge_polys[i] = sh.Polygon(np.vstack([top_lefts[i], top_rights[i], bottom_rights[i], bottom_lefts[i]]))

    s = sh.STRtree(edge_polys)

    areas = 0
    # compare each node with each other edge
    for i in range(len(node_polys)):
        res = s.query(node_polys[i], predicate = 'intersects')

        for j in res:
            curr_area = node_polys[i].intersection(edge_polys[j]).area / (node_polys[i].area + edge_polys[j].area)
            areas += curr_area

    ne = 1 - (areas / (n * m))

    return ne


"""
Edge-edge occlusion of the current layout

Input
coords:     np.ndarray, a 2xn numpy array containing the x,y coordinates
gtds:       np.ndarray, an nxn numpy array containing the graph theoretic distances (shortest path lengths)
r:          float, a value indicating the radius of a drawn node
width:      float, a value indicating the width of a drawn edge

Output
ee:         float, the edge-edge occlusion
"""


def edge_edge_occl(coords, gtds, r, width):

    # construct a graph from the shortest path matrix
    adj_matrix = copy.deepcopy(gtds)
    adj_matrix[adj_matrix > 1] = 0
    graph = nx.from_numpy_array(adj_matrix)
    graph = nx.convert_node_labels_to_integers(graph)

    edges = list(graph.edges())
    m = len(edges)

    edges = np.array(edges)
    # get the y and x coordinates of the edges in their separate arrays
    # subtract
    y2s = coords[edges[:,1]][:,1]
    y1s = coords[edges[:,0]][:,1]
    x2s = coords[edges[:,1]][:,0]
    x1s = coords[edges[:,0]][:,0]
    angles = np.arctan2(y2s - y1s, x2s - x1s)

    half_width = width / 2

    # calculate the offset for the corners
    offsets = half_width * np.array([-np.sin(angles), np.cos(angles)])

    r_off = r * np.array([np.cos(angles), np.sin(angles)])

    # find the corner points
    top_lefts = (np.array([x1s, y1s]) - offsets + r_off).T
    top_rights = (np.array([x2s, y2s]) - offsets - r_off).T
    bottom_lefts = (np.array([x1s, y1s]) + offsets + r_off).T
    bottom_rights = (np.array([x2s, y2s]) + offsets - r_off).T

    rect_poly = [0] * m
    areas = 0

    # compare each edge with each other edge
    for i in range(m):
        rect_poly[i] = sh.Polygon(np.vstack([top_lefts[i], top_rights[i], bottom_rights[i], bottom_lefts[i]]))

    s = sh.STRtree(rect_poly)

    for i in range(len(rect_poly)):
        res = s.query(rect_poly[i], predicate = 'intersects')

        for j in res:
            # exclude self intersects
            if i != j:
                curr_area = rect_poly[i].intersection(rect_poly[j]).area / (rect_poly[i].area + rect_poly[j].area)
                areas += curr_area

    ee = 1 - (areas / (m**2 - m))

    return ee


"""
Angular resolution deviation of the current layout

Input
coords:     np.ndarray, a 2xn numpy array containing the x,y coordinates
gtds:       np.ndarray, an nxn numpy array containing the graph theoretic distances (shortest path lengths)

Output
ar:         float, the angular resolution deviation
"""


def angular_resolution_dev(coords, gtds):

    adj_matrix = copy.deepcopy(gtds)
    adj_matrix[adj_matrix > 1] = 0
    graph = nx.from_numpy_array(adj_matrix)
    graph = nx.convert_node_labels_to_integers(graph)

    # initialize variables
    n = graph.number_of_nodes()
    nodes = list(graph.nodes())

    all_angles = []
    # loop over all nodes
    for i in range(n):
        # only compute angles if there are at least 2 edges to a node
        curr_degree = graph.degree(nodes[i])
        smallest_angle = 360
        if curr_degree > 1:
            best_angle = 360 / curr_degree
            curr_neighbs = list(graph.neighbors(nodes[i]))

            # get the ordering and then get the angles of that specific ordering
            order_neighbs = compute_order(curr_node = nodes[i], neighbors = curr_neighbs, coords = coords)
            norm_sub = np.subtract(coords[order_neighbs, ].copy(), coords[nodes[i], ])
            sub_phi = (np.arctan2(norm_sub[:, 1:2], norm_sub[:, :1]) * 180 / np.pi)
            # get the degrees to positive 0-360
            sub_phi = ((sub_phi + 360) % 360).flatten()


            # compare the last edge with the first edge
            first = sub_phi[0]
            last = sub_phi[-1]
            angle = abs(first - last)

            # if the angle is smaller than 360 then save that as the new angle
            if angle < smallest_angle:
                smallest_angle = angle

            # now compare each consecutive edge pair to get the smallest seen angle
            while len(sub_phi) >= 2:
                first = sub_phi[0]
                second = sub_phi[1]

                # if the angle is smaller than 360 then save that as the new angle
                angle = abs(first - second)
                if angle < smallest_angle:
                    smallest_angle = angle

                sub_phi = np.delete(sub_phi, 0)

            # add the deviation of the smallest angle to the ideal angle to a list
            all_angles.append(abs((best_angle - smallest_angle) / best_angle))

    ar = 1 - np.mean(all_angles)

    return ar


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
