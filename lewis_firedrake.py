from firedrake import *
import math

# --- Mesh ---
ncells = 50
L = 8 * math.pi
H = 10.0
nlayers = 50

base_mesh = PeriodicIntervalMesh(ncells, L)
mesh = ExtrudedMesh(base_mesh, layers=nlayers, layer_height=H / nlayers)

x, v = SpatialCoordinate(mesh)
mesh.coordinates.interpolate(as_vector([x, v - H / 2]))

# --- Function spaces ---
V    = FunctionSpace(mesh, 'DQ', 1)
Wbar = FunctionSpace(mesh, 'CG', 1, vfamily='R', vdegree=0)

# --- Initial condition ---
fn = Function(V, name="density")
fn.interpolate(v**2 * exp(-v**2 / 2) * (1 + 0.05 * cos(0.5 * x)) / (2 * math.pi)**0.5)

One  = Function(V).assign(1.0)
fbar = assemble(fn * dx) / assemble(One * dx)

# --- Poisson solver for electrostatic potential ---
phi   = Function(Wbar, name="potential")
fstar = Function(V)

psi  = TestFunction(Wbar)
dphi = TrialFunction(Wbar)

phi_eqn   = dphi.dx(0) * psi.dx(0) * dx - H * (fstar - fbar) * psi * dx
shift_eqn = dphi.dx(0) * psi.dx(0) * dx + dphi * psi * dx

nullspace   = VectorSpaceBasis(constant=True, comm=COMM_WORLD)
phi_problem = LinearVariationalProblem(lhs(phi_eqn), rhs(phi_eqn), phi, aP=shift_eqn)
phi_solver  = LinearVariationalSolver(
    phi_problem,
    nullspace=nullspace,
    solver_parameters={'ksp_type': 'gmres', 'pc_type': 'lu', 'ksp_rtol': 1e-8},
)

# --- DG advection solver ---
dtc    = Constant(0)
df_out = Function(V)

q  = TestFunction(V)
df = TrialFunction(V)
n  = FacetNormal(mesh)
u  = as_vector([v, -phi.dx(0)])
un = 0.5 * (dot(u, n) + abs(dot(u, n)))

df_a = q * df * dx
df_L = dtc * (
    div(u * q) * fstar * dx
    - (q('+') - q('-')) * (un('+') * fstar('+') - un('-') * fstar('-')) * (dS_h + dS_v)
    - conditional(dot(u, n) > 0, 1, 0) * q * dot(u, n) * fstar * ds_tb
)

df_problem = LinearVariationalProblem(df_a, df_L, df_out)
df_solver  = LinearVariationalSolver(df_problem)

# --- Time stepping (SSPRK3) ---
T      = 50.0
nsteps = 5000
dt     = T / nsteps
ndump  = 100
dtc.assign(dt)

f1 = Function(V)
f2 = Function(V)

outfile = VTKFile("vlasov.pvd")

fstar.assign(fn)
phi_solver.solve()
outfile.write(fn, phi)
phi.assign(0.0)

dumpn = 0
for step in ProgressBar("Timestep").iter(range(nsteps)):

    fstar.assign(fn)
    phi_solver.solve()
    df_solver.solve()
    f1.assign(fn + df_out)

    fstar.assign(f1)
    phi_solver.solve()
    df_solver.solve()
    f2.assign(0.75 * fn + 0.25 * (f1 + df_out))

    fstar.assign(f2)
    phi_solver.solve()
    df_solver.solve()
    fn.assign(fn / 3 + 2 * (f2 + df_out) / 3)

    dumpn += 1
    if dumpn % ndump == 0:
        dumpn = 0
        outfile.write(fn, phi)
