from new import circle_intpl as new
from old import circle_intpl as old

import numpy as np
from numpy.testing import assert_allclose
import random
from scipy.spatial.transform import Rotation as R


def sample_circle(origin: np.ndarray, normal: np.ndarray, radius: float, N=10) -> np.ndarray:
    """ Sample N points of a circle centered at origin with the given normal at the given radius"""
    # Find vector perpendicular to the normal
    v = get_orthogonal_vector(normal)
    p = []
    for _ in range(N):
        # Choose random angle
        angle = random.random() * 2 * np.pi
        # Rotate the vector we constructed in the beginning by angle degrees
        # around the normal vector
        rot = R.from_rotvec(normal * angle).as_matrix()
        v = rot @ v

        # Calculate the point on the boundary along the newly rotated vector
        p.append(origin + v * radius)
    return np.array(p)


def get_orthogonal_vector(n):
    if n[2] < n[0]:
        v = np.array([n[1], -n[0], 0], dtype=float)
    else:
        v = np.array([0, -n[2], n[1]], dtype=float)

    # Normalize
    v /= np.linalg.norm(v)
    return v


def random_vector() -> np.ndarray:
    x = random.random()
    y = random.random()
    z = random.random()
    return np.array([x, y, z])


def normalize(v) -> np.ndarray:
    return v / np.linalg.norm(v)


def perturb_along_vector(point: np.ndarray, vector: np.ndarray) -> np.ndarray:
    t = random.random() * 2 - 1
    return point + vector * 0.01 * t


def test(points, origin, normal):
    new_normal, new_origin = new(points)
    old_normal, old_origin = old(points)

    assert_allclose(new_normal, old_normal)
    assert_allclose(new_origin, old_origin)
    assert_allclose(new_origin, origin, err_msg="Origin did not match")
    assert_allclose(old_origin, origin, err_msg="Origin did not match")

    if np.all(normal > 0) or np.all(normal < 0):

        assert_allclose(np.abs(new_normal), np.abs(
            normal), err_msg="Normals did not match")
        assert_allclose(np.abs(old_normal), np.abs(
            normal), err_msg="Normals did not match")


# Test old vs new
for i in range(1000):
    # Setup: Create a random circle and perturb it
    normal = normalize(random_vector())
    radial = normalize(get_orthogonal_vector(normal))
    origin = random_vector()
    radius = random.random() * 10
    points = sample_circle(origin, normal, radius)
    print(f"Running test {i}:", end=" ")
    test(points, origin, normal)
    print(f"Passed!")
print("Passed all tests!")
