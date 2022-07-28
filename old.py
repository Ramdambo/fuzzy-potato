import numpy as np
import numpy.matlib as mtlb
import math

def QuatToRot(q0, q1, q2, q3):
    R = [[2 * q0 * q0 - 1 + 2 * q1 * q1, 2 * q1 * q2 - 2 * q0 * q3, 2 * q1 * q3 + 2 * q0 * q2],
    [2 * q1 * q2 + 2 * q0 * q3, 2 * q0 * q0 - 1 + 2 * q2 * q2, 2 * q2 * q3 - 2 * q0 * q1],
    [2 * q1 * q3 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 2 * q0 * q0 - 1 + 2 * q3 * q3]]
    return R

def plotcirc3d(n1, n2, n3, ur1, ur2, ur3, radius, ax):
    n = [n1, n2, n3]
    phi = math.atan2(n2, n1)
    u = np.zeros(3)
    u[0] = -math.sin(phi)
    u[1] = math.cos(phi)
    Ur = [ur1[0], ur2[0], ur3[0]]
    pcircle = np.zeros([math.ceil(2 * math.pi / 0.02), 3])
    ax.scatter(Ur[0], Ur[1], Ur[2])
    ax.plot([Ur[0], Ur[0] + n[0]], [Ur[1], Ur[1] + n[1]], [Ur[2], Ur[2] + n[2]])
    i = 0
    for t in np.arange(0, 2 * math.pi, 0.02):
        pcircle[i, :] = (radius * math.cos(t) * u) + radius * math.sin(t) * np.cross(n, u) + Ur
        i = i + 1
    ax.plot(pcircle[:, 0], pcircle[:, 1], pcircle[:, 2])

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def circle_intpl(points):
    nr = len(points)
    meanp = np.mean(points, axis = 0)
    meanp_mat = mtlb.repmat(meanp, nr, 1)
    A = points - meanp_mat
    U, S, V = np.linalg.svd(A)
    n = V[2, :]
    z = [0, 0, 1]
    k = np.cross(n, z)
    nk = np.linalg.norm(k)
    k = np.divide(k, nk)
    wos = np.matmul(np.transpose(n), z)
    theta = math.acos(wos)
    q0 = math.cos(theta / 2)
    q1 = k[0] * math.sin(theta / 2)
    q2 = k[1] * math.sin(theta / 2)
    q3 = k[2] * math.sin(theta / 2)
    R = QuatToRot(q0, q1, q2, q3)
    Prot = np.zeros([nr, len(A[0])])
    for i in range(0, nr):
        Prot[i] = np.transpose(A[i][:]) - np.multiply(np.matmul(A[i][:], n), n)
    Prot = np.transpose(Prot)
    Prot1 = Prot
    Prot = np.matmul(R, Prot1)
    AA = np.ones([3, nr])
    AA[0:2] = Prot[0:2]
    AA = np.transpose(AA)
    b = np.zeros([nr, 1])
    for i in range(0, nr):
        b[i][0] = AA[i][0] * AA[i][0] + AA[i][1] * AA[i][1]
    c = np.matmul(np.linalg.pinv(AA), b)
    xc = c[0] / 2
    yc = c[1] / 2
    radius = math.sqrt(c[2] + xc * xc + yc * yc)
    Ur = np.matmul(np.transpose(R), np.array([xc, yc, 0], dtype = object)) + meanp
    Prot1 = np.transpose(Prot1) + mtlb.repmat(meanp, nr, 1)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(Prot1[:, 0], Prot1[:, 1], Prot1[:, 2])
    points = np.array(points)
    up = np.cross(points[0, :] - [Ur[0][0], Ur[1][0], Ur[2][0]], points[3, :] - [Ur[0][0], Ur[1][0], Ur[2][0]])
    angle = angle_between((up[0], up[1], up[2]), (n[0], n[1], n[2]))
    if np.abs(angle) > 1.57:
        n = -n
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    # plotcirc3d(n[0], n[1], n[2], Ur[0], Ur[1], Ur[2], radius, ax)
    # ax.plot([Ur[0][0], Ur[0][0] + up[0]], [Ur[1][0], Ur[1][0] + up[1]], [Ur[2][0], Ur[2][0] + up[2]])
    # ax.scatter(points[0, 0], points[0, 1], points[0, 2])
    # plt.show()
    return n, Ur

# P = [[0.408, -0.208, 1.015], [0.524, -0.217, 1.291], [0.537, -0.213, 1.535], [0.431, -0.221, 1.78]]

# circle_intpl(P)