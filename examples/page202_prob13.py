"""Solution to problem 13 from page 202 of Axler's Linear Algebra Done Right"""
import sympy

from pylinearalg import BasisVector, Vector, VectorSpace, gram_schmidt


def to_poly(u: Vector):
    return sympy.Poly(u.components, sympy.symbols('x'))


def inner_product(p, q):
    """Evaluate the definite integral from -pi to pi of pq."""
    x = sympy.symbols('x')
    if isinstance(p, Vector):
        p = to_poly(p)
    if isinstance(q, Vector):
        q = to_poly(q)
    return (p * q).as_expr().integrate((x, -sympy.pi, sympy.pi))


if __name__ == '__main__':
    dim = 6  # defines dimension of our space, P_dim(R)
    basis = [BasisVector(f"x^{i}") for i in range(dim)]
    V = VectorSpace(basis, inner_product)

    initial_list = [Vector([1 if j == i else 0 for j in range(dim)], space=V) for i in range(dim)]
    result_list = gram_schmidt(initial_list)
    for e in result_list:
        print(sympy.latex(to_poly(e)))

    x = sympy.symbols('x')
    sin_appx = sympy.Poly(0, x)  # starts as 0
    for e in result_list:
        sin_appx += to_poly(e.scaled(inner_product(sympy.sin(x), e)))  # project onto e

    print(sympy.latex(sin_appx))
