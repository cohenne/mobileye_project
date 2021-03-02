import numpy as np
from math import sqrt


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)

    if abs(tZ) < 10e-6:
        print('tz = ', tZ)
    elif norm_prev_pts.size == 0:
        print('no prev points')
    elif norm_curr_pts.size == 0:
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ)
    
    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))

    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []

    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)

        if not valid:
            Z = 0

        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)

    return corresponding_ind, np.array(pts_3D), validVec


def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point
    return (pts - pp) / focal


def unnormalize(pts, focal, pp):
    # transform normalized pixels into pixels using the focal length and principle point
    return pts * focal + pp


def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    R = EM[:3, :3]
    T = EM[:3, 3]
    tZ = T[2]
    foe = np.array([T[0]/tZ, T[1]/tZ])

    return R, foe, tZ


def rotate(pts, R):
    # rotate the points - pts using R
    for i in range(len(pts)):
        normalizePt = np.array([pts[i][0], pts[i][1], 1])
        rotatePt = R.dot(normalizePt)
        pts[i][0] = rotatePt[0] / rotatePt[2]
        pts[i][1] = rotatePt[1] / rotatePt[2]

    return pts


def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index
    m = (foe[1] - p[1]) / (foe[0] - p[0])
    n = (p[1]*foe[0] - foe[1]*p[0]) / (foe[0] - p[0])

    def epipolar_line(x):
       return  m * x + n

    def distance(y1, y2):
        return abs( (y1 - y2) / sqrt(m*m + 1) )

    minDiffrence = 2
    minInd = 0

    for inedx in range(len(norm_pts_rot)):
        y = m * norm_pts_rot[inedx][0] + n
        dis = distance(norm_pts_rot[inedx][1], y)

        if dis < minDiffrence:
            minDiffrence = dis
            minInd = inedx

    return minInd, norm_pts_rot[minInd]


def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    # combine the two estimations and return estimated Z
    
    Zx = ( tZ * (foe[0] - p_rot[0]) ) / (p_curr[0] - p_rot[0])
    Zy = ( tZ * (foe[1] - p_rot[1]) ) / (p_curr[1] - p_rot[1])

    x_diff = abs(p_rot[0] - p_curr[0])
    y_diff = abs(p_rot[1] - p_curr[1])

    return (x_diff/(x_diff+y_diff))*Zx + (y_diff/(x_diff+y_diff))*Zy


