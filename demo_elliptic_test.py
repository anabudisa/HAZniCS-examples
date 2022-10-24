"""
\file HAZniCS_examples/demo_elliptic_test.py
Created by Miroslav Kuchta, Ana Budisa.
We solve

    u + div(grad(u)) = f

on a unit square or unit cube domain with homogeneous Neumann BC.
The preconditioner for this system is AMG (hazmath or hypre).
Outer solver is Conjugate Gradients.
"""
from dolfin import *
from block.iterative import ConjGrad
from block.algebraic.hazmath import AMG as AMG_haz
from block.algebraic.petsc.precond import AMG as AMG_petsc
import haznics

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-dim', default=3, choices=(2, 3), type=int)  # dimension of the problem
parser.add_argument('-N', default=8, type=int)  # n for unit square/cube mesh
parser.add_argument('-refine', default=5, type=int)  # number of refinements (starts from N and goes 2*N, 4*N, 8*N, etc)
args = parser.parse_args()

results_haz = {'ndof': {}, 'iter': {}, 'Tsetup': {}, 'Tsolve': {}, 'Ttotal': {}}
results_petsc = {'ndof': {}, 'iter': {}, 'Tsetup': {}, 'Tsolve': {}, 'Ttotal': {}}

for i in range(args.refine):
    n = args.N * 2**i
    mesh = {2: UnitSquareMesh,
            3: UnitCubeMesh}[args.dim](*(n, )*args.dim)

    V = FunctionSpace(mesh, "CG", 1)

    results_haz['ndof'][i] = V.dim()
    results_petsc['ndof'][i] = V.dim()

    f = Expression("sin(pi*(x[0]))", degree=2)
    u, v = TrialFunction(V), TestFunction(V)

    a = u*v*dx + dot(grad(u), grad(v))*dx
    L = f*v*dx

    A = assemble(a)
    b = assemble(L)

    if args.dim < 3:
        # choose some other parameters
        params_haz = {}
        params_petsc = {}
    else:
        # choices for hazmath AMG parameters
        params_haz = {
                    "print_level": 2,                       # (0, 1, 2, 4, 8, 10), 0 - nothing, 10 - everything
                    "AMG_type": haznics.UA_AMG,             # (UA, SA) + _AMG
                    "cycle_type": haznics.NL_AMLI_CYCLE,    # (V, W, AMLI, NL_AMLI, ADD) + _CYCLE
                    "smoother": haznics.SMOOTHER_GS,        # SMOOTHER_ + (JACOBI, GS, SGS, CG, SOR, SSOR, GSOR, SGSOR, POLY, L1DIAG, FJACOBI, FGS, FSGS)
                    "coarse_solver": 32,                    # (32 = SOLVER_UMFPACK, 0 = ITERATIVE)
                    "aggregation_type": haznics.VMB,        # (VMB, MIS, MWM, HEC)
                    "strong_coupled": 0.0,                  # threshold for dropping values
                    "max_aggregation": 100,
                    }

        # choices for hypre parameters
        params_petsc = {
                    "pc_hypre_boomeramg_cycle_type": "W",           # (V,W)
                    "pc_hypre_boomeramg_truncfactor": 0.3,          # Truncation factor for interpolation (between 0 and 1)
                    "pc_hypre_boomeramg_agg_nl": 2,                 # Number of levels of aggressive coarsening (0-1 for 2d, 2-5 for 3d)
                    "pc_hypre_boomeramg_strong_threshold": 0.5,     # Threshold for being strongly connected (.25 for 2d and .5 for 3d)
                    "pc_hypre_boomeramg_coarsen_type": "HMIS",      # (Ruge-Stueben, modifiedRuge-Stueben, Falgout, PMIS, HMIS)
                    "pc_hypre_boomeramg_interp_type": "ext+i-cc",   # (classical, direct, multipass, multipass-wts, ext+i, ext+i-cc, standard, standard-wts, FF, FF1)
                    "pc_hypre_boomeramg_print_statistics": 0,
                    "pc_hypre_boomeramg_print_debug": 0,
                    }

    # setup preconditioners
    # from hazmath preconds
    B_haz = AMG_haz(A, parameters=params_haz)
    print("*" * 70)
    # from block/petsc
    B_petsc = AMG_petsc(A, parameters=params_petsc)

    # setup solvers
    Ainv_haz = ConjGrad(A, precond=B_haz, tolerance=1e-6, show=10)
    Ainv_petsc = ConjGrad(A, precond=B_petsc, tolerance=1e-6, show=10)

    # solve
    print("hazmath" + "*" * 70)
    x1 = Ainv_haz*b
    results_haz['iter'][i] = len(Ainv_haz.residuals)
    results_haz['Tsetup'][i] = B_haz.setup_time
    results_haz['Tsolve'][i] = Ainv_haz.cputime
    results_haz['Ttotal'][i] = Ainv_haz.cputime + B_haz.setup_time

    print("hypre" + "*" * 70)
    x2 = Ainv_petsc*b
    results_petsc['iter'][i] = len(Ainv_petsc.residuals)
    results_petsc['Tsetup'][i] = B_petsc.setup_time
    results_petsc['Ttotal'][i] = Ainv_petsc.cputime
    results_petsc['Tsolve'][i] = Ainv_petsc.cputime - B_petsc.setup_time

    # plot results
    u1 = Function(V)
    u1.vector()[:] = x1[:]
    # plot(u1, title="u, computed by hazmath [x=Ainv*b]")
    File("solution.pvd") << u1

    # u2 = Function(V)
    # u2.vector()[:] = x2[:]
    # plot(u2, title="u, computed by petsc [x=Ainv*b]")

print(results_haz)
print(results_petsc)

# save results to file
with open('results.txt', 'w') as file:
    file.write('--------------- hazmath ---------------\n')
    row = r'%s \\' + '\n'
    keyz = results_haz.keys()
    for i in range(args.refine):
        res = [results_haz[key][i] for key in keyz]
        file.write(row % (' & '.join(map(str, res))))
    file.write("---------------------------------------\n")

    file.write('\n--------------- hypre ---------------\n')
    keyz = results_petsc.keys()
    for i in range(args.refine):
        res = [results_petsc[key][i] for key in keyz]
        file.write(row % (' & '.join(map(str, res))))
    file.write("---------------------------------------\n")
