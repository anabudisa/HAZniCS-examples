"""
\file examples/haznics/demo_darcy_stokes.py
Created by Miroslav Kuchta, Ana Budisa on 2021-07-23.

We solve on [0, 1]^2. Darcy domain is [0.25, 0.75]^2 while Stokes is its 
complement with repsect to the unit square.

(Stokes)
-div(T(u1, p1)) = f1
        div(u1) = 0

with T(u1, p1) = -p1*I + mu*grad(u1)

(Darcy)
K^-1 u2 + grad(p2) = f2
           div(u2) = 0

(coupling)
u1*n1 + u2*n2 = gD
   -(T.n1).n1 = p2 + gN
  -(T.n1).tau = alpha*uD.tau + gT

The coupling is enforced by Lagrange multiplier. The preconditioner for this
system is based on Riesz map wrt inner product
mu*H^1 x (1/mu)*L^2 x (1/K)*(I-grad div) x K*L^2 x ((1/mu)*H^{-0.5} + K*H^{0.5}).
"""
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
    

def setup_mms(mparams):
    """Data for method of manufactured solutions' test case"""
    mesh = UnitSquareMesh(2, 2)  # Dummy
    x, y = SpatialCoordinate(mesh)
    
    V = FunctionSpace(mesh, 'CG', 2)
    # Define constants as function to allow ufly substition
    mu, K, alpha = Constant(1), Constant(1), Constant(1)

    # Auxiliary function for defining Stokes velocity
    phi = sin(pi * (x + y))  # Aux expr

    p1 = sin(4 * pi * x) * cos(pi * y)
    p2 = cos(2 * pi * x) * cos(2 * pi * y)
    
    # Now the Stokes velocity is
    u1 = as_vector((phi.dx(1), -phi.dx(0)))  # To be divergence free

    # Stokes stress
    stokes_stress = lambda u, p: -p * Identity(2) + mu*grad(u)
    stokes_traction = lambda n, u=u1, p=p1: dot(n, stokes_stress(u, p))  # Vector
    # Forcing for Stokes
    f1 = -div(stokes_stress(u1, p1))

    # Darcy pressure, velocity and force
    u2 = -K * grad(p2)
    f2 = div(u2)

    n = Constant((1, 0))
    # Coupling data, for piece of boundary
    gD = dot(u1, n) - dot(u2, n)
    gN = -p2 - dot(n, stokes_traction(n, u1, p1))
    gT = -tangent(stokes_traction(n, u1, p1), n) + alpha*tangent(u1, n)

    mu_, K_, alpha_ = sp.symbols('mu K alpha')
    subs = {mu: mu_, K: K_, alpha: alpha_}

    as_expression = lambda f: ulfy.Expression(f, subs=subs, degree=4,
                                              mu=mparams.mu,
                                              K=mparams.K,
                                              alpha=mparams.alpha)

    data = {'solution': [as_expression(f) for f in (u1, p1, u2, p2, -dot(n, stokes_traction(n)))],
            'vol_f': [as_expression(f) for f in (f1, f2)],
            'iface_f': [as_expression(f) for f in (gD, gN, gT)],
            'dirichlet': [as_expression(f) for f in (u1, u2)],
            'neumann': [as_expression(f) for f in (stokes_stress(u1, p1), p2)]}

    return data


def setup_domains(n, mms):
    '''Tag cell and facet functions of the meshes involved'''
    mesh = UnitSquareMesh(n, n)
    #   4       4 
    # 1   2   1   2
    #   3       3
    subdomains = MeshFunction('size_t', mesh, 2, 2)
    CompiledSubDomain('x[0] < 0.5+DOLFIN_EPS').mark(subdomains, 1)

    # Stokes
    mesh1 = EmbeddedMesh(subdomains, 1)
    # Tag it
    bdries1 = MeshFunction('size_t', mesh1, mesh1.topology().dim() - 1, 0)
    DomainBoundary().mark(bdries1, 3)
    CompiledSubDomain('near(x[0], 0)').mark(bdries1, 1)
    CompiledSubDomain('near(x[0], 0.5)').mark(bdries1, 2)

    # Darcy
    mesh2 = EmbeddedMesh(subdomains, 2)
    # Tag it
    bdries2 = MeshFunction('size_t', mesh2, mesh2.topology().dim() - 1, 0)
    DomainBoundary().mark(bdries2, 3)    
    CompiledSubDomain('near(x[0], 0.5)').mark(bdries2, 1)
    CompiledSubDomain('near(x[0], 1.0)').mark(bdries2, 2)

    # And interface
    bmesh = EmbeddedMesh(bdries1, 2)   

    return {'darcy': bdries2, 'stokes': bdries1, 'interface': bmesh}


def get_system(n, family, mparams, data):
    """Setup the linear system A*x = b in W where W has bcs"""
    K, mu, alpha = Constant(mparams.K), Constant(mparams.mu), Constant(mparams.alpha)

    domains = setup_domains(n, data)
    bdries1, bdries2, bmesh = domains['stokes'], domains['darcy'], domains['interface']
    mesh1, mesh2 = bdries1.mesh(), bdries2.mesh()
    lm_tags = (1, )

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
        h1 = avg(FacetArea(mesh1))
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

    f1, f2 = data['vol_f']
    gD, gN, gT = data['iface_f']
    # Data for dirichlet bc is only
    u1_true = data['dirichlet'][0]  # Strongly on top and bottom
    # Data for neumann bcs; For stokes we set stress on left and for
    # darcy the pressure is imposed everywhere
    sigma1_true, p2_true = data['neumann']  # 

    # For Neuamnn bcs we'll need
    n1, n2 = FacetNormal(mesh1), FacetNormal(mesh2)

    ds1 = Measure('ds', domain=mesh1, subdomain_data=bdries1)
    ds2 = Measure('ds', domain=mesh2, subdomain_data=bdries2)

    L = block_form(W, 1)

    L[0] = (inner(f1, v1) * dx +
            inner(dot(sigma1_true, n1), v1) * ds1(3) + 
            inner(gT, tangent(Tv1, n_)) * dx_)

    # Darcy forcing has the contrib due to coupling for normal tractions
    L[2] = (-inner(gN, dot(Tv2, n_)) * dx_ -
            inner(p2_true, dot(v2, n2)) * ds2(2))

    L[3] = -inner(f2, q2) * dx
    # Coupling of velocities
    L[4] = inner(gD, q) * dx_

    A, b = map(ii_assemble, (a, L))

    V1_bcs = [DirichletBC(V1, u1_true, bdries1, 1)]
    V2_bcs = [DirichletBC(V2, u2_true, bdries2, 3)]
    bcs = [V1_bcs, [], V2_bcs, [], []]

    A, b = apply_bc(A, b, bcs)

    return A, b, W, bcs


def get_preconditioner_blocks(AA, W, bcs, mparams, mms):
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
    L2 = inner(Constant((0, 0)), v2)*dx
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


def matrix_power(A, M, s, scale=1.):
    '''Power by solving GEVP'''
    A_, M_ = scale*A.array(), scale*M.array()
    lmbda, U = scipy.linalg.eigh(A_, M_)

    W = np.dot(M_, U)
    Lmbda = np.diag(lmbda**s)

    return np.dot(np.dot(W, Lmbda), W.T)


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


def exact_riesz_map_preconditioner(AA, W, bcs, mparams, mms):
    '''All blocks inverted exactly'''
    blocks = get_preconditioner_blocks(AA, W, bcs, mparams=mparams, mms=mms)

    B0, B1, B2, B3 = blocks[:4]
    # Setup fractional part
    A, M = blocks[-1]
    mu, K = mparams.mu, mparams.K
    
    B4 = PETScMatrix(
        PETSc.Mat().createDense(
            size=(A.size(0), A.size(1)), 
            array=matrix_power(A, M, s=-0.5, scale=1./mu) + matrix_power(A, M, s=0.5, scale=K)
        )
    )
    # Now apply LU everywhere
    return block_diag_mat([petsc.LU(Bi) for Bi in (B0, B1, B2, B3, B4)])


def exact_riesz_map_preconditioner_wRA(AA, W, bcs, mparams, mms):
    '''All blocks by LU exact fractional part by RA'''
    blocks = get_preconditioner_blocks(AA, W, bcs, mparams=mparams, mms=mms)

    B0, B1, B2, B3 = blocks[:4]
    BB_blocks = [petsc.LU(Bi) for Bi in (B0, B1, B2, B3)]
    
    A, M = blocks[-1]    
    BB_blocks.append(get_rational_preconditioner(A, M, mparams))

    return block_diag_mat(BB_blocks)


def inexact_riesz_map_preconditioner_wRA(AA, W, bcs, mparams, mms):
    '''All blocks are realized by multivel approach'''
    blocks = get_preconditioner_blocks(AA, W, bcs, mparams=mparams, mms=mms)

    B0, B1, B2, B3 = blocks[:4]

    BB_blocks = [petsc.AMG(B0),
                 petsc.AMG(B1),
                 petsc.HypreAMS(B2, W[2]),
                 petsc.AMG(B3)]
    
    A, M = blocks[-1]    
    BB_blocks.append(get_rational_preconditioner(A, M, mparams))

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
    parser.add_argument('-K_value', type=float, default=2., help='K in Darcy')
    parser.add_argument('-mu_value', type=float, default=3., help='mu in Stokes')
    parser.add_argument('-alpha_value', type=float, default=0.5, help='alpha on Interface')    
    # For debugging it is useful with direct solver
    parser.add_argument('-direct_solver', type=int, default=0, help='Use direct solver?')

    parser.add_argument('-elm_family', type=str, default='TH', choices=['TH', 'CR'])
    
    # Iterative solver
    parser.add_argument('-preconditioner', type=str, choices=('exact', 'exactRA', 'inexactRA'), default='exact',
                        help='Realization of inverse in the preconditioner')
    args, _ = parser.parse_known_args()
    
    # Parameters
    mparams = namedtuple('Parameters', ('mu', 'K', 'alpha'))(args.mu_value, args.K_value, args.alpha_value)
    print(f'Parameters: {mparams}')

    # Setup a MMS
    data = setup_mms(mparams)
    u1_true, p1_true, u2_true, p2_true, p_true = data['solution']

    get_preconditioner = {'exact': exact_riesz_map_preconditioner,
                          'exactRA': exact_riesz_map_preconditioner_wRA,
                          'inexactRA': inexact_riesz_map_preconditioner_wRA}[args.preconditioner]
    
    headers_perf = ['ndofs', 'ndofs_i', 'niters', 'time', '|r|']
    headers_error = ['h', '|euS|_1', '|epS|_0', '|euD|_div', '|epD|_0', '|ep|_0']
    table_perf, table_error = [], []


    result_dir = './results'
    not os.path.exists(result_dir) and os.mkdir(result_dir)

    result_path = f'ds2D_flat_K{args.K_value}_mu{args.mu_value}_alpha{args.alpha_value}_precond{args.preconditioner}_family{args.elm_family}.txt'
    result_path = os.path.join(result_dir, result_path)

    with open(result_path, 'w') as out:
        out.write('# %s\n' % ' '.join(headers_perf + headers_error))
    
    errors0, h0 = None, None
    for k in range(args.nrefs):
        n = 4*2**k
        # Setup system
        AA, bb, W, bcs = get_system(n, family=args.elm_family, mparams=mparams, data=data)

        wh = ii_Function(W)                
        # Solve
        if args.direct_solver:
            then = time.time()            

            solver = LUSolver('umfpack')
            solver.solve(ii_convert(AA), wh.vector(), ii_convert(bb))
            dt = time.time() - then
            niters = -1

            r_norm = (AA*wh.block_vec()-bb).norm()
        else:
            then = time.time()            
            # Hazmath rational approximation preconditioner
            BB = get_preconditioner(AA, W, bcs,
                                    mparams=mparams, mms=data)

            cbk = lambda k, x, r, b=bb, A=AA: print(f'\titer{k} -> {[(b[i]-xi).norm("l2") for i, xi in enumerate(A*x)]}')            
            AAinv = MinRes(AA, precond=BB, tolerance=1E-12, show=4)#, callback=cbk)
            xx = AAinv * bb

            dt = time.time() - then

            for i, xxi in enumerate(xx):
                wh[i].vector().axpy(1, xxi)
            niters = len(AAinv.residuals)
            r_norm = AAinv.residuals[-1]

        h = wh[0].function_space().mesh().hmin()
        ndofs = sum(Wi.dim() for Wi in W)
        ndofs_i = W[-1].dim()
        
        euS = errornorm(u1_true, wh[0], 'H1', degree_rise=2)
        epS = errornorm(p1_true, wh[1], 'L2', degree_rise=2)
        euD = errornorm(u2_true, wh[2], 'Hdiv', degree_rise=2)
        epD = errornorm(p2_true, wh[3], 'L2', degree_rise=2)
        ep = errornorm(p_true, wh[4], 'L2', degree_rise=2)        
        errors = np.array([euS, epS, euD, epD, ep])

        if errors0 is None:
            rates = [np.nan]*len(errors)
        else:
            rates = np.log(errors/errors0)/np.log(h/h0)
        errors0, h0 = errors, h

        errors_rates = tuple(f'{e:.3E}[{r:.2f}]' for (e, r) in zip(errors, rates))
        
        row_ = (ndofs, ndofs_i, niters, dt, r_norm)
        table_perf.append(row_)

        row = (h, ) + errors_rates
        table_error.append(row)

        print(tabulate.tabulate(table_perf, headers=headers_perf))
        print(tabulate.tabulate(table_error, headers=headers_error))        

        with open(result_path, 'a') as out:
            item = row_ + (h, ) + tuple(errors)
            out.write('%s\n' % (' '.join(tuple(map(str, item)))))
