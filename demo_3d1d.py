"""
\file HAZniCS-examples/demo_3d1d.py
Created by Miroslav Kuchta, Ana Budisa.

We solve on Omega (3d) and Gamma (1d) a reduced EMI problem of signal propagation in neurons,
that is the steady-state electrodiffusion in a porous tissue (3d) and the network of 1d curves (immersed in Omega)

    - div(sigma_3 grad(p3)) + (rho * C_m / deltat) * (Avg(p3) - p1) * delta_Gamma = f3 on Omega
            - div(rho^2 sigma_1 grad(p1)) + (rho * C_m / deltat) * (p1 - Avg(p3)) = f1 on Gamma

with Avg(p3) the average of the function on Omega to Gamma over a cylinder-type surface
of rho-radius around Gamma. delta_Gamma is the delta-distribution on Gamma.
sigma_3 and sigma_1 are extra- and intracellular conductivities. C_m is the membrane capacitance parameter.
deltat is the time step size (this linear system results from a time-stepping scheme for the dynamic diffusion problem)

We enforce homogeneous Neumann conditions on the outer boundary of the 3d (and 1d) domain.
We solve the problem with Conjugate Gradient method preconditioned with "metric AMG" method that
uses block Schwarz smoothers.
"""
from scipy.sparse import csr_matrix
from xii.assembler.average_matrix import average_matrix as average_3d1d_matrix, trace_3d1d_matrix
from block.algebraic.hazmath import block_mat_to_block_dCSRmat
from dolfin import *
from xii import *
import haznics
import time


def get_mesh_neuron():
    '''Load it'''
    mesh = Mesh()
    with HDF5File(mesh.mpi_comm(), './downloads/haznics-examples-meshes/P14_rat1_layerIII_cell1_0.h5', 'r') as h5:
        h5.read(mesh, '/mesh', False)
        edge_f = MeshFunction('size_t', mesh, 1, 0)
        h5.read(edge_f, '/neuron')

    return edge_f


def get_system(edge_f, k3=1e0, k1=1e0, gamma=1e0, coupling_radius=0.):
    """A, b, W, bcs"""
    assert edge_f.dim() == 1

    # Meshes
    meshV = edge_f.mesh()  #
    meshQ = EmbeddedMesh(edge_f, 1)

    # Spaces
    V = FunctionSpace(meshV, 'CG', 1)
    Q = FunctionSpace(meshQ, 'CG', 1)
    W = [V, Q]

    u, p = map(TrialFunction, W)
    v, q = map(TestFunction, W)

    # Average (coupling_radius > 0) or trace (coupling_radius = 0)
    if coupling_radius > 0:
        # Averaging surface
        cylinder = Circle(radius=coupling_radius, degree=10)
        Ru, Rv = Average(u, meshQ, cylinder), Average(v, meshQ, cylinder)
        C = average_3d1d_matrix(V, Q, cylinder)
    else:
        Ru, Rv = Average(u, meshQ, None), Average(v, meshQ, None)
        C = trace_3d1d_matrix(V, Q, meshQ)

    # Line integral
    dx_ = Measure('dx', domain=meshQ)

    # Parameters
    k3, k1, gamma = map(Constant, (k3, k1, gamma))
    f3, f1 = Expression('x[0] + x[1]', degree=4), Constant(1)

    # We're building a 2x2 problem
    a = block_form(W, 2)
    a[0][0] = k3 * inner(grad(u), grad(v)) * dx + k3 * inner(u, v) * dx
    a[1][1] = k1 * inner(grad(p), grad(q)) * dx + k1 * inner(p, q) * dx

    m = block_form(W, 2)
    m[0][0] = inner(Ru, Rv) * dx_
    m[0][1] = -inner(p, Rv) * dx_
    m[1][0] = -inner(q, Ru) * dx_
    m[1][1] = inner(p, q) * dx_

    L = block_form(W, 1)
    L[0] = inner(f3, v) * dx
    L[1] = inner(f1, q) * dx

    AD, M, b = map(ii_assemble, (a, m, L))

    # Coupling info
    C = csr_matrix(C.getValuesCSR()[::-1], shape=C.size)

    return (AD, M), b, W, C


def solve_haznics(W, A, b, AD, M, C):
    def block_to_haz(AA):
        # first make sure the whole matrix is of block_mat type
        if hasattr(AA, 'block_collapse'):
            AA = AA.block_collapse()

        # then make sure each block is a petsc matrix
        brow, bcol = AA.blocks.shape
        for i in range(brow):
            for j in range(bcol):
                AA[i][j] = ii_collapse(AA[i][j])

        AAhaz = block_mat_to_block_dCSRmat(AA)

        return AAhaz

    dimW = sum([VV.dim() for VV in W])
    start_time = time.time()
    # convert vectors
    bb = ii_convert(b)
    b_np = bb[:]
    bhaz = haznics.create_dvector(b_np)
    xhaz = haznics.dvec_create_p(dimW)

    # convert matrices
    Ahaz = block_to_haz(A)
    Mhaz = block_to_haz(M)
    ADhaz = block_to_haz(AD)
    # coupling incidence matrix C
    csr0, csr1, csr2 = C.indptr, C.indices, C.data
    Chaz = haznics.create_matrix(csr2, csr1, csr0, C.shape[1])
    # print("\n------------------- Data conversion time: ", time.time() - start_time, "\n")

    # call solver
    niters = haznics.fenics_metric_amg_solver(Ahaz, bhaz, xhaz, ADhaz, Mhaz, Chaz)

    return niters, xhaz

# --------------------------------------------------------------------


if __name__ == '__main__':
    import numpy as np
    # Load mesh
    edge_f = get_mesh_neuron()

    # Parameters
    sigma3d, sigma1d = 3e0, 7e0  # conductivities in mS cm^-1 (from EMI book, Buccino paper)
    mc = 1  # membrane capacitance in microF cm^-2
    radius = 5  # radius (rho) of the averaging surface in micro m
    deltat_inv = 1e2  # inverse of the time step, in s^-1 ( 1/dt )

    gamma = deltat_inv * 2 * np.pi * radius * mc  # coupling parameter
    sigma1d = sigma1d * np.pi * radius**2  # scaled 1d conductivity

    # Get discrete system
    start_time = time.time()
    (AD, M), b, W, C = get_system(edge_f, k3=sigma3d, k1=sigma1d, gamma=gamma, coupling_radius=radius)
    A = AD + gamma * M
    print("\n------------------ System setup and assembly time: ", time.time() - start_time, "\n")

    # Solve (CG + AMG)
    niters, xhaz = solve_haznics(W, A, b, AD, M, C)

    # Results
    dimV, dimQ = W[0].dim(), W[1].dim()
    print("************************")
    print("Parameters: ", f'{sigma3d=}, {sigma1d=}, {radius=}, {deltat_inv=}', "\n")
    print(f'dim(V)={dimV} dim(Q)={dimQ}  hmax(V)={W[0].mesh().hmax():.2f}  hmin(V)={W[0].mesh().hmin():.2f}  hmin(Q)={W[1].mesh().hmin():.2f} '
          f'niters={niters}')
    print("************************")

    # Post-process: export to vtu
    xx = xhaz.to_ndarray()
    wh = ii_Function(W)
    wh[0].vector().set_local(xx[:dimV])
    wh[1].vector().set_local(xx[dimV:])

    File('neuron_sol3d.pvd') << wh[0]
    File('neuron_sol1d.pvd') << wh[1]

