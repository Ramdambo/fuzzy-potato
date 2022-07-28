import numpy as np
from scipy.spatial.transform import Rotation

# Refactored by Daniel Gusenburger (d.gusenburger@zema.de)

# Requirements:
#   pip install numpy scipy
# OR:
#   conda install numpy scipy
# depending on what you use


def normalize(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Calculate angle between v1 and v2, clipped to pi """
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def circle_intpl(points):
    meanp = np.mean(points, axis=0, keepdims=True)

    A = points - meanp
    B = A @ A.T

    # Singular value decomposition to fit plane through points
    *_, V = np.linalg.svd(A)
    # Fitted normal vector
    n = V[2, :]

    # Need to rotate plane such that normal vectors align
    # ( 0 | 0 | 1 ) x n = ( n[1] | -n[0] | 0 )
    axis = normalize(np.array([n[1], -n[0], 0]))
    # ( 0 | 0 | 1 ) . n = n[2]
    theta = np.arccos(n[2])

    # No need to use quaternion as in the paper, we already have the axis and
    # the angle, just convert from rotation vector
    R = Rotation.from_rotvec(theta * axis).as_matrix()

    # Project points to fitted plane
    proj = A - (A @ n).reshape(10, 1) @ n.reshape(1, 3)
    # Rotate plane about axis by angle theta such that normal aligns with
    # world normal
    proj = R @ proj.T

    # Solve linear system of equations
    b = np.diag(B)
    c = np.linalg.pinv(proj.T) @ b

    xc = c[0] / 2
    yc = c[1] / 2

    # Reproject origin
    origin = R.T @ [xc, yc, 0] + meanp.flatten()

    up = np.cross(points[0] - origin, points[3] - origin)
    angle = angle_between(up, n)

    if np.abs(angle) > np.pi / 2:
        n = -n

    return n, origin
