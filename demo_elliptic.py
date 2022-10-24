"""
\file HAZniCS_examples/demo_elliptic.py
Created by Ana Budisa.
Example copyright by bitbucket.org/fenics-apps/cbc.block.

We solve

    u + div(grad(u)) = f

on a unit square domain with homogeneous Neumann BC.
The preconditioner for this system is UA-AMG (from HAZmath).
Outer solver is Conjugate Gradients.
"""
from block.iterative import ConjGrad
from block.algebraic.hazmath import AMG
from dolfin import *

# Function spaces, elements
mesh = UnitCubeMesh(32, 32, 32)

V = FunctionSpace(mesh, "CG", 1)
u, v = TrialFunction(V), TestFunction(V)
f = Expression("sin(pi*x[0])", degree=2)

a = inner(u, v) * dx + inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx

A = assemble(a)
b = assemble(L)

# here we use hazmath AMG
B = AMG(A, parameters={"max_levels": 10, "AMG_type": 1})
Ainv = ConjGrad(A, precond=B, tolerance=1e-10, show=2)

# solve
x = Ainv * b

u = Function(V)
u.vector()[:] = x[:]

# default solver in Dolfin 
u2 = Function(V)
solve(A, u2.vector(), b)

print("Max differences between the two solutions: ", (u.vector() - u2.vector()).max())

