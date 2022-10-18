from xii import *
import block.algebraic.petsc as petsc
from block.algebraic.hazmath import RA, AMG, HXDiv
import ulfy, haznics, scipy
from petsc4py import PETSc
from dolfin import *
import sympy as sp


def tangent(v, n):
    '''Tangent part'''
    return v - n*dot(v, n)
    

def get_system(family, mparams):
    """Setup the linear system A*x = b in W where W has bcs"""
    K, mu, alpha = Constant(mparams.K), Constant(mparams.mu), Constant(mparams.alpha)

    mesh = Mesh()
    with HDF5File(mesh.mpi_comm(), './downloads/haznics-examples-meshes/brain_mesh.h5', 'r') as f:
        f.read(mesh, '/mesh', False)
        
        tdim = mesh.topology().dim()
        subdomains = MeshFunction('size_t', mesh, tdim, 0)
        interfaces = MeshFunction('size_t', mesh, tdim-1, 0)
        
        f.read(subdomains, '/subdomains')
        f.read(interfaces, '/boundaries')

    File('interfaces.pvd') << interfaces
    File('subdomains.pvd') << subdomains
    exit()
    lm_tags = (5, 6)
    # Stokes - SAS, ventricles, CP
    mesh1 = EmbeddedMesh(subdomains, (1, 3, 4))
    bdries1 = mesh1.translate_markers(interfaces, lm_tags)

    bdries1_ = MeshFunction('size_t', mesh1, mesh1.topology().dim()-1, 0)
    CompiledSubDomain('on_boundary and x[2] > -35 - tol', tol=1E-10).mark(bdries1_, 1)
    CompiledSubDomain('on_boundary and x[2] <= -35 + tol', tol=1E-10).mark(bdries1_, 2)

    bdries1.array()[np.where(np.logical_and(bdries1_.array() == 1,
                                            ~np.logical_or(bdries1.array() == 5, bdries1.array() == 6)))] = 1
    bdries1.array()[np.where(np.logical_and(bdries1_.array() == 2,
                                            ~np.logical_or(bdries1.array() == 5, bdries1.array() == 6)))] = 2
    
    mesh2 = EmbeddedMesh(subdomains, (2, ))
    bdries2 = mesh2.translate_markers(interfaces, lm_tags)

    bmesh = EmbeddedMesh(bdries1, lm_tags)

    if family == 'TH':
        V1 = VectorFunctionSpace(mesh1, 'CG', 2)
        Q1 = FunctionSpace(mesh1, 'CG', 1)
    else:
        V1 = VectorFunctionSpace(mesh1, 'CR', 1)
        Q1 = FunctionSpace(mesh1, 'DG', 0)
    # Darcy
    V2 = FunctionSpace(mesh2, 'RT', 1)
    Q2 = FunctionSpace(mesh2, 'DG', 0)
    # The multiplier
    Q = FunctionSpace(bmesh, 'DG', 0)

    W = [V1, Q1, V2, Q2, Q]
    print([Wi.dim() for Wi in W])
    
    u1, p1, u2, p2, p = map(TrialFunction, W)
    v1, q1, v2, q2, q = map(TestFunction, W)
    # Stokes traces
    Tu1, Tv1 = Trace(u1, bmesh), Trace(v1, bmesh)
    # Darcy traces
    Tu2, Tv2 = Trace(u2, bmesh), Trace(v2, bmesh)

    # The line integral and orient iface
    dx_ = Measure('dx', domain=bmesh)
    n_ = OuterNormal(bmesh, orientation=mesh1)

    a = block_form(W, 2)
    # Stokes
    a[0][0] = inner(mu * grad(u1), grad(v1)) * dx + alpha * inner(tangent(Tv1, n_), tangent(Tu1, n_)) * dx_
    if family == 'CR':
        h1 = sqrt(avg(FacetArea(mesh1)))
        gamma1 = Constant(5)
        a[0][0] += (mu*gamma1/h1)*inner(jump(u1), jump(v1))*dS
    
    a[0][1] = -inner(p1, div(v1)) * dx
    a[0][4] = inner(p, dot(Tv1, n_)) * dx_
    # Darcy
    a[2][2] = K ** -1 * inner(u2, v2) * dx
    a[2][3] = -inner(p2, div(v2)) * dx
    a[2][4] = -inner(p, dot(Tv2, n_)) * dx_
    # Symmetrize
    a[1][0] = -inner(q1, div(u1)) * dx
    a[3][2] = -inner(q2, div(u2)) * dx

    a[4][0] = inner(q, dot(Tu1, n_)) * dx_
    a[4][2] = -inner(q, dot(Tu2, n_)) * dx_

    f1, f2 = Constant((0, 0, 0)), Constant(0)

    ds1 = Measure('ds', domain=mesh1, subdomain_data=bdries1)
    ds2 = Measure('ds', domain=mesh2, subdomain_data=bdries2)

    L = block_form(W, 1)

    r = SpatialCoordinate(mesh1)
    c = Constant((0.047, 0, 0))
    r = (r-c)/sqrt(np.dot(r-c, r-c))
    
    n1 = FacetNormal(mesh1)
    L[0] = inner(f1, v1)*dx - inner(dot(r, r)*Constant(1E-2), dot(v1, n1))*ds1(2)

    L[3] = -inner(f2, q2) * dx

    A, b = map(ii_assemble, (a, L))
    
    V1_bcs = [DirichletBC(V1, Constant((0, 0, 0)), bdries1, 1)]
    V2_bcs = []
    bcs = [V1_bcs, [], V2_bcs, [], []]

    A, b = apply_bc(A, b, bcs)

    return A, b, W, bcs


def get_preconditioner_blocks(AA, W, bcs, mparams):
    """Discrete operators needed to get the preconditioner"""
    K, mu, alpha = Constant(mparams.K), Constant(mparams.mu), Constant(mparams.alpha)

    V1, Q1, V2, Q2, Q = W

    # Use grad-grad to control the Stokes velocity
    B0 = AA[0][0]

    # Stokes pressure is
    p1, q1 = TrialFunction(Q1), TestFunction(Q1)
    B1 = assemble((1 / mu) * inner(p1, q1) * dx)

    # Darcy flux
    u2, v2 = TrialFunction(V2), TestFunction(V2)
    b2 = (1 / K) * (inner(u2, v2) * dx + inner(div(u2), div(v2)) * dx)
    L2 = inner(Constant((0, 0, 0)), v2)*dx
    B2, _ = assemble_system(b2, L2, bcs[2])
    #mesh2 = V2.mesh()
    #h2 = avg(FacetArea(mesh2))
    #gamma2 = Constant(5)
    #n2 = FacetNormal(mesh2)
    #b2 += ((1/K)*gamma2/h2)*inner(jump(u2, n2), jump(v2, n2))*dS
    #B2 = assemble(b2)

    # Darcy pressure
    p2, q2 = TrialFunction(Q2), TestFunction(Q2)
    B3 = assemble(K * inner(p2, q2) * dx)

    # Multiplier
    p, q = TrialFunction(Q), TestFunction(Q)

    if Q.ufl_element().family() == 'Discontinuous Lagrange':
        bmesh = Q.mesh()
        # assert Q.ufl_element().degree() == 0
        h = CellDiameter(bmesh)
        h_avg = avg(h)

        a = h_avg ** (-1) * dot(jump(p), jump(q)) * dS + inner(p, q) * dx
    else:
        a = inner(grad(p), grad(q)) * dx + inner(p, q) * dx
    m = inner(p, q) * dx
    A, M = map(assemble, (a, m))

    blocks = [B0, B1, B2, B3, (A, M)]

    return blocks


def get_rational_preconditioner(A, M, mparams):
    """Realize inv((1/mu)*H^{-0.5} + K*H^{-0.5}) by RA"""

    mu, K = mparams.mu, mparams.K
    # parameters for RA and AMG
    params = {'coefs': [1. / mu, K], 'pwrs': [-0.5, 0.5],  # for RA
              'print_level': 0,
              'AMG_type': haznics.SA_AMG,
              'cycle_type': haznics.V_CYCLE,
              "max_levels": 20,
              "tol": 1E-10,
              "smoother": haznics.SMOOTHER_GS,
              "relaxation": 1.2,  # Relaxation in the smoother
              "coarse_dof": 10,
              "aggregation_type": haznics.VMB,  # (VMB, MIS, MWM, HEC)
              "strong_coupled": 0.0,  # threshold
              "max_aggregation": 100
              }

    return RA(A, M, parameters=params)


def inexact_riesz_map_preconditioner_wRA(AA, W, bcs, mparams):
    '''All blocks are realized by multivel approach'''
    blocks = get_preconditioner_blocks(AA, W, bcs, mparams=mparams)

    B0, B1, B2, B3 = blocks[:4]

    B2_params = {'Schwarz_levels': 0, 'print_level': 2}

    B0_parameters = {
        'AMG_type': haznics.SA_AMG,
        'cycle_type': haznics.V_CYCLE,
        "max_levels": 20,
        "tol": 1E-10,
        "smoother": haznics.SMOOTHER_GS,
        "relaxation": 1.2,  # Relaxation in the smoother
        "coarse_dof": 10,
        "aggregation_type": haznics.VMB,  # (VMB, MIS, MWM, HEC)
        "strong_coupled": 0.0,  # threshold
        "max_aggregation": 100
    }
        
    BB_blocks = [AMG(B0, parameters=B0_parameters),
                 petsc.AMG(B1),
                 HXDiv(B2, W[2], parameters=B2_params),
                 petsc.LU(B3)]
    
    
    A, M = blocks[-1]    
    BB_blocks.append(get_rational_preconditioner(A, M, mparams))

    return block_diag_mat(BB_blocks)


def inexact_riesz_map_preconditioner_wRA_Layton(AA, W, bcs, mparams):
    '''All blocks are realized by multivel approach'''
    blocks = get_preconditioner_blocks(AA, W, bcs, mparams=mparams)

    B0, B1, B2, B3 = blocks[:4]

    B2_params = {'Schwarz_levels': 0, 'print_level': 2}

    B0_parameters = {
        'AMG_type': haznics.SA_AMG,
        'cycle_type': haznics.V_CYCLE,
        "max_levels": 20,
        "tol": 1E-10,
        "smoother": haznics.SMOOTHER_GS,
        "relaxation": 1.2,  # Relaxation in the smoother
        "coarse_dof": 10,
        "aggregation_type": haznics.VMB,  # (VMB, MIS, MWM, HEC)
        "strong_coupled": 0.0,  # threshold
        "max_aggregation": 100
    }
        
    BB_blocks = [AMG(B0, parameters=B0_parameters),
                 petsc.AMG(B1),
                 HXDiv(B2, W[2], parameters=B2_params),
                 petsc.LU(B3)]
    
    
    A, M = blocks[-1]    
    BB_blocks.append(get_rational_preconditioner(A, M, MParams(1E12, mparams.K, mparams.alpha)))

    return block_diag_mat(BB_blocks)



def inexact_riesz_map_preconditioner_L2(AA, W, bcs, mparams):
    '''All blocks are realized by multivel approach'''
    blocks = get_preconditioner_blocks(AA, W, bcs, mparams=mparams)

    B0, B1, B2, B3 = blocks[:4]

    B2_params = {'Schwarz_levels': 0, 'print_level': 2}

    B0_parameters = {
        'AMG_type': haznics.SA_AMG,
        'cycle_type': haznics.V_CYCLE,
        "max_levels": 20,
        "tol": 1E-10,
        "smoother": haznics.SMOOTHER_GS,
        "relaxation": 1.2,  # Relaxation in the smoother
        "coarse_dof": 10,
        "aggregation_type": haznics.VMB,  # (VMB, MIS, MWM, HEC)
        "strong_coupled": 0.0,  # threshold
        "max_aggregation": 100
    }
        
    BB_blocks = [AMG(B0, parameters=B0_parameters),
                 petsc.AMG(B1),
                 HXDiv(B2, W[2], parameters=B2_params),
                 petsc.LU(B3)]
    
    
    A, M = blocks[-1]    
    BB_blocks.append(petsc.LU(ii_convert(mparams.K*M)))

    return block_diag_mat(BB_blocks)

# --------------------------------------------------------------------

if __name__ == '__main__':
    from block.iterative import MinRes
    from collections import namedtuple
    import argparse, time, tabulate
    import numpy as np
    import os
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-nrefs', type=int, default=1, help='Number of mesh refinements')
    # Material properties
    parser.add_argument('-K_value', type=float, default=1E-4, help='K in Darcy')
    parser.add_argument('-mu_value', type=float, default=3., help='mu in Stokes')
    parser.add_argument('-alpha_value', type=float, default=0.5, help='alpha on Interface')    
    # For debugging it is useful with direct solver
    parser.add_argument('-direct_solver', type=int, default=0, help='Use direct solver?')

    parser.add_argument('-elm_family', type=str, default='CR', choices=['TH', 'CR'])            
    # Iterative solver
    parser.add_argument('-preconditioner', type=str, choices=('inexactRA', 'inexactRA_Layton', 'mass'), default='inexactRA',
                        help='Realization of inverse in the preconditioner')
    args, _ = parser.parse_known_args()
    
    # Parameters
    MParams = namedtuple('Parameters', ('mu', 'K', 'alpha'))
    mparams = MParams(args.mu_value, args.K_value, args.alpha_value)
    print(f'Parameters: {mparams}')

    get_preconditioner = {
        'mass': inexact_riesz_map_preconditioner_L2,
        'inexactRA_Layton': inexact_riesz_map_preconditioner_wRA_Layton,
        'inexactRA': inexact_riesz_map_preconditioner_wRA
    }[args.preconditioner]
    
    headers_perf = ['ndofs', 'ndofs_i', 'niters', 'time', '|r|', 'h']
    table_perf = []

    result_dir = './results'
    not os.path.exists(result_dir) and os.mkdir(result_dir)

    result_path = f'ds3D_brain_K{args.K_value}_mu{args.mu_value}_alpha{args.alpha_value}_precond{args.preconditioner}_family{args.elm_family}.txt'
    result_path = os.path.join(result_dir, result_path)

    with open(result_path, 'w') as out:
        out.write('# %s\n' % ' '.join(headers_perf))
    
    # Setup system
    AA, bb, W, bcs = get_system(family=args.elm_family, mparams=mparams)
    
    wh = ii_Function(W)                

    then = time.time()            
    # Hazmath rational approximation preconditioner
    BB = get_preconditioner(AA, W, bcs, mparams=mparams)

    true_res_history = []
    def cbk(k, x, r, b=bb, A=AA, B=BB):
        errors = [(b[i]-xi).norm("l2") for i, xi in enumerate(A*x)]
        true_res_history.append(errors)
        print(f'\titer{k} -> {errors}')
    AAinv = MinRes(AA, precond=BB, tolerance=1E-10, show=4, maxiter=500, callback=cbk)
    xx = AAinv * bb
    
    dt = time.time() - then

    for i, xxi in enumerate(xx):
        wh[i].vector().axpy(1, xxi)
        # File(f'brain_sub{i}_.pvd') << wh[i]
    niters = len(AAinv.residuals)
    r_norm = AAinv.residuals[-1]

    h = wh[0].function_space().mesh().hmin()
    ndofs = sum(Wi.dim() for Wi in W)
    ndofs_i = W[-1].dim()
        
    row_ = (ndofs, ndofs_i, niters, dt, r_norm)
    table_perf.append(row_)

    print(tabulate.tabulate(table_perf, headers=headers_perf))

    with open(result_path, 'a') as out:
        item = row_ + (h, ) 
        out.write('%s\n' % (' '.join(tuple(map(str, item)))))

        np.savetxt(out, np.c_[np.array(true_res_history), AAinv.residuals[:-1]])
