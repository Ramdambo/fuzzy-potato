import numpy as np
import numpy.matlib as mtlb
import math


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    # https://en.wikipedia.org/wiki/Rotation_matrix
    x, y, z = axis
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    cos = np.cos(angle)
    sin = np.sin(angle)

    return np.array(
        [
            [cos + xx * (1 - cos), xy * (1 - cos) - z * sin, xz * (1 - cos) + y * sin],
            [xy * (1 - cos) + z * sin, cos + yy * (1 - cos), yz * (1 - cos) - x * sin],
            [xz * (1 - cos) - y * sin, yz * (1 - cos) + x * sin, cos + zz * (1 - cos)],
        ]
    )



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
        pcircle[i, :] = (
            (radius * math.cos(t) * u) + radius * math.sin(t) * np.cross(n, u) + Ur
        )
        i = i + 1
    ax.plot(pcircle[:, 0], pcircle[:, 1], pcircle[:, 2])


def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

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
    # points: Nx3
    nr = len(points)

    # Centroid 
    meanp = np.mean(points, axis=0)
    meanp_mat = mtlb.repmat(meanp, nr, 1)

    A = points - meanp_mat

    # Singular value decomposition to fit plane through points
    U, S, V = np.linalg.svd(A)

    # Normal vector of plane fitted through points
    n = V[2, :]

    half_theta = math.acos(n[2]) / 2

    # Construct rotation matrix:
    # Rotate by half_theta around normal vector
    R = rotation_matrix(n, half_theta)

    Prot = np.zeros((nr, len(A[0])))

    for i in range(0, nr):
        Prot[i] = A[i].T - np.multiply(np.matmul(A[i], n), n)

    Prot = np.matmul(R, Prot.T)

    AA = Prot.T

    b = np.zeros((nr, 1))

    for i in range(0, nr):
        b[i][0] = AA[i][0] * AA[i][0] + AA[i][1] * AA[i][1]

    c = np.matmul(np.linalg.pinv(AA), b)

    xc = c[0] / 2
    yc = c[1] / 2

    # TODO: Here we could have a problem since I used keepdims=True in meanp to eliminate meanp_mat
    # so essentially meanp == meanp_mat
    Ur = np.matmul(R.T, np.array([xc, yc, 0])) + meanp

    up = np.cross(
        points[0] - Ur[:, 0],
        points[3] - Ur[:, 0]
    )

    angle = angle_between(up, n)

    if np.abs(angle) > 1.57:
        n = -n

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(Prot1[:, 0], Prot1[:, 1], Prot1[:, 2])
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    # plotcirc3d(n[0], n[1], n[2], Ur[0], Ur[1], Ur[2], radius, ax)
    # ax.plot([Ur[0][0], Ur[0][0] + up[0]], [Ur[1][0], Ur[1][0] + up[1]], [Ur[2][0], Ur[2][0] + up[2]])
    # ax.scatter(points[0, 0], points[0, 1], points[0, 2])
    # plt.show()
    return n, Ur


# P = [[0.408, -0.208, 1.015], [0.524, -0.217, 1.291], [0.537, -0.213, 1.535], [0.431, -0.221, 1.78]]
# circle_intpl(P)
