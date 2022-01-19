"""Solution to problem 5 from page 190 of Axler's Linear Algebra Done Right"""
import sympy

from pylinearalg import BasisVector, Vector, VectorSpace, gram_schmidt


def inner_product(p, q):
    """Evaluate the definite integral from 0 to 1 of pq."""
    total = sympy.Mul(0)
    for exp1, coef1 in enumerate(p.components):
        for exp2, coef2 in enumerate(q.components):
            # integral of (coef1)x^exp1 + (coef2)x^exp2 is:
            total += coef1 * coef2 / (exp1 + exp2 + 1)
    return total


basis = [BasisVector("1"), BasisVector("x"), BasisVector("x²")]
V = VectorSpace(basis, inner_product)

u = Vector([1, 0, 0], space=V)
v = Vector([0, 1, 0], space=V)
w = Vector([0, 0, 1], space=V)

for vec in gram_schmidt([u, v, w]):
    print(vec.normalized())
