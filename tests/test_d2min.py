import numpy as np
from numpy.linalg import inv

import pytest

def get_d2min_m1(b0, b):
    """ Calculates D2min for a set of bonds
    Args
        b0: initial bond lengths
        b: final bond lengths
    """
    V = b0.transpose().dot(b0)
    W = b0.transpose().dot(b)
    J = inv(V).dot(W)
    non_affine = b0.dot(J) - b
    d2min = np.sum(np.square(non_affine))
    return d2min


def get_d2min_m2(b0, b):
    """ Calculates D2min for a set of bonds, implemented in the exact as it is
    shown in Falk & Langer's 1997 PRE

    Args
        b0: initial bond lengths
        b: final bond lengths
    """
    X = b.transpose().dot(b0)
    Y = b0.transpose().dot(b0)
    J = X.dot(inv(Y.transpose()))
    non_affine = b0.dot(J.transpose()) - b
    d2min = np.sum(np.square(non_affine))
    return d2min

def test_d2min_same():

    b0 = np.array([[2], [3]])
    b = np.array([[6], [6]])

    ans1 = get_d2min_m1(b0, b)
    ans2 = get_d2min_m2(b0, b)
    ans = pytest.approx(np.sum(np.square(30/13*b0 - b)))
    assert ans1 == ans
    assert ans2 == ans

    b0 = np.array([[2, 1],
                   [3, 4],
                   [4, 5]])
    
    b = np.array([[6, 1],
                  [6, 0],
                  [4, 5]])

    ans1 = get_d2min_m1(b0, b)
    ans2 = get_d2min_m2(b0, b)
    sol = np.array([[ 3.74193548,  0.64516129],
                    [-1.83870968,  0.09677419]])
    ans = b0.dot(sol) - b
    ans = pytest.approx(np.sum(np.square(ans)))
    assert ans1 == ans
    assert ans2 == ans

    rng: np.random.Generator = np.random.default_rng(0)
    delta = 0.2
    b0 = rng.random((6, 2)) - 0.5
    b = b0 + (rng.random((6, 2)) - 0.5)*delta
    
    ans1 = get_d2min_m1(b0, b)
    ans2 = get_d2min_m2(b0, b)
    print(ans1, ans2)
    assert ans1 == pytest.approx(ans2)