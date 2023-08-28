import numpy as np
import copy
import networkx as nx
import time
import shapely as sh


def crossing_res(coords, gtds):
    curr_min_angle = 360

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

    s = sh.STRtree(polys)

    for i in polys:
        i_coords = i.xy
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

            slope1 = y_diff_1 / x_diff_1
            slope2 = y_diff_2 / x_diff_2
            angle = abs((slope2 - slope1) / (1 + slope1 * slope2))
            deg_angle = np.arctan(angle) * 180 / np.pi

            if deg_angle < curr_min_angle:
                curr_min_angle = deg_angle

    if curr_min_angle == 360:
        cross_res = 1
    else:
        cross_res = curr_min_angle / 90

    return cross_res


def crossings_number(coords, gtds):

    cnt = 0

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

    s = sh.STRtree(polys)

    for i in polys:
        res = s.query(i, predicate = 'crosses')
        cnt += len(res)

    return 1 - (cnt / (m**2))


def norm_stress(coords, gtds, stress_alpha = 2):

    with np.errstate(divide = 'ignore'):
        # compute the weights and set the numbers that turned to infinity (the 0 on the diagonals) to 0
        weights = np.array(gtds).astype(float) ** -stress_alpha
        weights[weights == float('inf')] = 0

    # calculate the euclidean distances
    eucl_dis = np.sqrt(np.sum(((np.expand_dims(coords, axis = 1) - coords) ** 2), 2))
    scal_coords = coords * (np.nansum((eucl_dis / gtds) / np.nansum((eucl_dis ** 2) / (gtds ** 2))))

    eucl_dis_new = np.sqrt(np.sum(((np.expand_dims(scal_coords, axis=1) - scal_coords) ** 2), 2))

    # compute stress by multiplying the distances with the graph theoretic distances, squaring it, multiplying by the weight factor and then taking the sum
    stress_tot = np.sum(weights * ((eucl_dis_new - gtds) ** 2))
    ns = 1 - (stress_tot / (np.shape(coords)[0] ** 2))

    return ns


def edge_lengths_sd(coords, gtds):

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

    el = np.sqrt(np.sum((edge_dis - mu)**2) / m)

    return el


# def neighborhood_preservation(coords, gtds):
#
#     # get the euclidean distances
#     eucl_dis = np.sqrt(np.sum(((np.expand_dims(coords, axis=1) - coords) ** 2), 2))
#
#     intersect = (eucl_dis * gtds)
#     union = (eucl_dis + gtds).clip(0, 1)
#
#     if intersect.sum() == 0:
#         nep = 0
#     else:
#         nep = intersect.sum() / union.sum()
#
#     return nep


def node_resolution(coords):

    # calculate the euclidean distances
    eucl_dis = np.sqrt(np.sum(((np.expand_dims(coords, axis = 1) - coords) ** 2), 2))

    # exclude zeros
    eucl_dis = eucl_dis[np.nonzero(eucl_dis)]
    # ratio between smallest and largest eucl distance
    nr = np.min(eucl_dis) / np.max(eucl_dis)

    return nr


def node_resolution_old(coords, tar_res):

    # calculate the euclidean distances
    eucl_dis = np.sqrt(np.sum(((np.expand_dims(coords, axis = 1) - coords) ** 2), 2))

    # node resolution computation
    # exclude zeros
    eucl_dis = eucl_dis[np.nonzero(eucl_dis)]
    # find values smaller than target resolution
    mins = eucl_dis[eucl_dis < tar_res]

    # .any() included in case there are no values smaller than target res
    if mins.any():
        nr = np.mean(eucl_dis[eucl_dis < tar_res] / tar_res)
    else:
        nr = 1

    return nr

def node_node_occl(coords, r = 0.01):

    # get the number of nodes
    n = np.shape(coords)[0]

    # get the euclidean distance
    eucl_dis = np.sqrt(np.sum(((np.expand_dims(coords, axis=1) - coords) ** 2), 2))

    # get a mask for circles that are within the sum of radii, these are the nodes that overlap
    intersection_mask = np.less(eucl_dis, (2*r))

    # get the angle theta between the line connecting the centers and the x-axis
    dx = np.subtract.outer(coords[:, 0], coords[:, 0])
    dy = np.subtract.outer(coords[:, 1], coords[:, 1])
    theta = np.arctan2(dy, dx)

    # get the intersection area of the two nodes
    intersection_area = np.where(intersection_mask, r**2 * (theta - np.sin(theta)) + r**2 * (np.pi - theta), 0)

    # total area of a circle
    total_area = 2 * np.pi * r**2

    # get the intersection ratio
    intersection_ratio = (intersection_area / total_area)

    # nodes that perfectly overlap should be set to 1
    intersection_ratio[intersection_ratio == 0.5] = 1
    # nodes overlapping themselves do not count
    np.fill_diagonal(intersection_ratio, 0)

    # sum up the ratios and divide by the worst case scenario where every node is stacked ontop (exluding themselves),
    #nn = np.sum(intersection_ratio) / (n**2 - n)

    n_intersects = len(intersection_ratio > 0)
    if n_intersects > 0:
        nn = 1 - (np.sum(intersection_ratio) / n_intersects)
    else:
        nn = 1

    return nn


def node_edge_occl(coords, gtds, r, width):

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
    y2s = coords[edges[:,1]][:,1]
    y1s = coords[edges[:,0]][:,1]
    x2s = coords[edges[:,1]][:,0]
    x1s = coords[edges[:,0]][:,0]
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

    areas = 0

    node_polys = [0] * n
    edge_polys = [0] * m
    # compare each node with each other edge
    for i in range(n):
        node_polys[i] = sh.Point(coords[i]).buffer(r)

    for i in range(m):
        edge_polys[i] = sh.Polygon(np.vstack([top_lefts[i], top_rights[i], bottom_rights[i], bottom_lefts[i]]))

    s = sh.STRtree(edge_polys)

    intersect_cnt = 0
    for i in node_polys:
        res = s.query(i, predicate='intersects')
        for j in res:
            curr_area = i.intersection(edge_polys[j]).area / (i.area + edge_polys[j].area)
            areas += curr_area
            if curr_area > 0:
                intersect_cnt += 1

    #ne = 1 - (areas / (n * m))

    if intersect_cnt > 0:
        ne = 1 - (areas / intersect_cnt)
    else:
        ne = 1

    return ne


def edge_edge_occl(coords, gtds, r, width):

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

    # Step 4: Calculate the offset for the corners
    offsets = half_width * np.array([-np.sin(angles), np.cos(angles)])

    r_off = r * np.array([np.cos(angles), np.sin(angles)])

    # Step 5: Find the corner points
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

    intersect_cnt = 0
    for i in rect_poly:
        res = s.query(i, predicate='intersects')
        for j in res:
            curr_area = i.intersection(rect_poly[j]).area / (i.area + rect_poly[j].area)
            areas += curr_area
            if curr_area > 0:
                intersect_cnt += 1

    #ee = 1 - (areas / (m**2))
    if intersect_cnt > 0:
        ee = 1 - (areas / intersect_cnt)
    else:
        ee = 1

    return ee


"""
A simple function for computing the angular resolution of the current layout

Input
final_g_dict:   dict, a dictionary containg the coordinates "coords", a graph object G "G", the graph theoretic distances "gtds" and the stress alpha "stress_alpha"

Output
res:            float, the angular resolution
"""


def angular_resolution(coords, gtds):

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

            total = 360
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
                total = total - angle

                sub_phi = np.delete(sub_phi, 0)

            # we forgot to compare the last edge with the first edge so we simply check the total angle left with the current min angle
            if total < angle_res:
                angle_res = total

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


def norm_stress_node_resolution(coords, gtds, tar_res, stress_alpha = 2):

    with np.errstate(divide = 'ignore'):
        # compute the weights and set the numbers that turned to infinity (the 0 on the diagonals) to 0
        weights = np.array(gtds).astype(float) ** -stress_alpha
        weights[weights == float('inf')] = 0

    # calculate the euclidean distances
    eucl_dis = np.sqrt(np.sum(((np.expand_dims(coords, axis = 1) - coords) ** 2), 2))

    # compute stress by multiplying the distances with the graph theoretic distances, squaring it, multiplying by the weight factor and then taking the sum
    stress_tot = np.sum(weights * ((eucl_dis - gtds) ** 2))
    ns = 1 - (stress_tot / (np.shape(coords)[0] ** 2))

    # node resolution computation
    # exclude zeros
    eucl_dis = eucl_dis[np.nonzero(eucl_dis)]
    # find values smaller than target resolution
    mins = eucl_dis[eucl_dis < tar_res]

    # .any() included in case there are no values smaller than target res
    if mins.any():
        nr = np.mean(eucl_dis[eucl_dis < tar_res] / tar_res)
    else:
        nr = 1

    return ns, nr


def edge_edge_occl_crossing_number_crossing_res(coords, gtds, r, width):

    adj_matrix = copy.deepcopy(gtds)
    adj_matrix[adj_matrix > 1] = 0
    graph = nx.from_numpy_array(adj_matrix)
    graph = nx.convert_node_labels_to_integers(graph)

    edges = list(graph.edges())
    m = len(edges)

    edges = np.array(edges)
    # get the y and x coordinates of the edges in their separate arrays
    # subtract
    y2s = coords[edges[:, 1]][:, 1]
    y1s = coords[edges[:, 0]][:, 1]
    x2s = coords[edges[:, 1]][:, 0]
    x1s = coords[edges[:, 0]][:, 0]
    angles = np.arctan2(y2s - y1s, x2s - x1s)

    half_width = width / 2

    # Step 4: Calculate the offset for the corners
    offsets = half_width * np.array([-np.sin(angles), np.cos(angles)])

    r_off = r * np.array([np.cos(angles), np.sin(angles)])

    # Step 5: Find the corner points
    top_lefts = (np.array([x1s, y1s]) - offsets + r_off).T
    top_rights = (np.array([x2s, y2s]) - offsets - r_off).T
    bottom_lefts = (np.array([x1s, y1s]) + offsets + r_off).T
    bottom_rights = (np.array([x2s, y2s]) + offsets - r_off).T

    rect_dict = {}
    line_dict = {}
    areas = 0
    cnt = 0
    curr_min_angle = 360

    # compare each edge with each other edge
    for i in range(m):
        cur_e = tuple(edges[i])
        if cur_e not in rect_dict:
            # construct a rectangle
            rect_dict[cur_e] = sh.Polygon(np.vstack([top_lefts[i], top_rights[i], bottom_rights[i], bottom_lefts[i]]))
            c1 = [(coords[cur_e[0]][0], coords[cur_e[0]][1]), (coords[cur_e[1]][0], coords[cur_e[1]][1])]

            # it can happen that the layout algorithm puts two nodes on the EXACT same position, we simply continue with the next edge
            if c1[0] != c1[1]:
                line_dict[cur_e] = sh.LineString(c1)

        for j in range(i + 1, m):
            cur_e_2 = tuple(edges[j])
            if cur_e_2 not in rect_dict:
                rect_dict[cur_e_2] = sh.Polygon(
                    np.vstack([top_lefts[j], top_rights[j], bottom_rights[j], bottom_lefts[j]]))

                c2 = [(coords[cur_e_2[0]][0], coords[cur_e_2[0]][1]), (coords[cur_e_2[1]][0], coords[cur_e_2[1]][1])]
                line_dict[cur_e_2] = sh.LineString(c2)

            # if the rectangles intersect then get the intersection area %
            if rect_dict[cur_e].intersects(rect_dict[cur_e_2]):
                areas += rect_dict[cur_e].intersection(rect_dict[cur_e_2]).area / rect_dict[cur_e].area

            if cur_e[0] not in cur_e_2 and cur_e[1] not in cur_e_2:
                if line_dict[cur_e].intersects(line_dict[cur_e_2]):
                    cnt += 1
                    line1_xy = line_dict[cur_e].xy
                    line2_xy = line_dict[cur_e_2].xy

                    y_diff_1 = line1_xy[1][1] - line1_xy[0][1]
                    x_diff_1 = line1_xy[1][0] - line1_xy[0][0]
                    y_diff_2 = line2_xy[1][1] - line2_xy[0][1]
                    x_diff_2 = line2_xy[1][0] - line2_xy[0][0]

                    slope1 = 0
                    slope2 = 0
                    if x_diff_1 != 0:
                        slope1 = y_diff_1 / x_diff_1
                    if x_diff_2 != 0:
                        slope2 = y_diff_2 / x_diff_2

                    angle = abs((slope2 - slope1) / (1 + slope1 * slope2))
                    deg_angle = np.arctan(angle) * 180 / np.pi

                    if deg_angle < curr_min_angle:
                        curr_min_angle = deg_angle

    ee = 1 - (areas / (m * (m - 1) / 2))
    cn = 1 - (cnt / (m * (m - 1) / 2))

    if curr_min_angle == 360:
        cr = 1
    else:
        cr = curr_min_angle / 90

    return ee, cn, cr