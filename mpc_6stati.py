
# MPC dinamico a 6 stati con discretizzazione di Eulero e TV-QP (cvxpy).
# Stati: x = [X, Y, phi, vx, vy, omega]
# Ingressi: u = [d, delta] (drive command ~ trazione, angolo sterzo)

import numpy as np
import cvxpy as cp

# Parametri veicolo 

Params = {
    "Cm1": 0.287, "Cm2": 0.0545,
    "Cr0": 0.0518, "Cr2": 0.00035,
    "Br": 3.3852, "Cr": 1.2691, "Dr": 0.1737,
    "Bf": 2.579, "Cf": 1.2,    "Df": 0.192,
    "m": 0.041, "Iz": 27.8e-6, "lf": 0.029, "lr": 0.033,
    "g": 9.81,
    
    "maxAlpha": 0.6,      # rad
    "vx_zero": 0.3,       # m/s soglia per evitare divisioni per 0 nelle slip angles
}

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

# Forze gomme (Pacejka semplificato) e long force
def tire_forces(x, u, p):
    """
    Calcola F_fy, F_ry, F_rx con slip angles e Pacejka.
    """
    X, Y, phi, vx, vy, omega = x
    d, delta = u

    # Protezione vx: evita divisioni per 0 nel calcolo degli angoli di deriva
    vx_eff = np.sign(vx) * max(abs(vx), p["vx_zero"])

    # Slip angles 
    # alpha_f = -atan((omega*lf + vy)/vx) + delta
    # alpha_r =  atan((omega*lr - vy)/vx)
    alpha_f = -np.arctan2(omega * p["lf"] + vy, vx_eff) + delta
    alpha_r =  np.arctan2(omega * p["lr"] - vy, vx_eff)

    # Clipping per robustezza 
    alpha_f = clamp(alpha_f, -p["maxAlpha"], p["maxAlpha"])
    alpha_r = clamp(alpha_r, -p["maxAlpha"], p["maxAlpha"])

    # Pacejka: Fy = D * sin(C * arctan(B * alpha))
    Fy_f = p["Df"] * np.sin(p["Cf"] * np.arctan(p["Bf"] * alpha_f))
    Fy_r = p["Dr"] * np.sin(p["Cr"] * np.arctan(p["Br"] * alpha_r))

    # Longitudinale posteriore
    # Frx = (Cm1 - Cm2*vx)*d - Cr0 - Cr2*vx^2
    Frx = (p["Cm1"] - p["Cm2"] * vx) * d - p["Cr0"] - p["Cr2"] * (vx**2)

    return Fy_f, Fy_r, Frx

# Dinamica continua f(x,u)
def f_cont(x, u, p):
    X, Y, phi, vx, vy, omega = x
    d, delta = u
    m, Iz, lf, lr = p["m"], p["Iz"], p["lf"], p["lr"]

    Fy_f, Fy_r, Frx = tire_forces(x, u, p)

    Xdot   = vx * np.cos(phi) - vy * np.sin(phi)
    Ydot   = vx * np.sin(phi) + vy * np.cos(phi)
    phidot = omega

    vxdot  = (1.0/m) * (Frx - Fy_f * np.sin(delta) + m * vy * omega)
    vydot  = (1.0/m) * (Fy_r - Fy_f * np.cos(delta) - m * vx * omega)
    omegadot = (1.0/Iz) * (Fy_f * lf * np.cos(delta) - Fy_r * lr)

    return np.array([Xdot, Ydot, phidot, vxdot, vydot, omegadot])

# Discretizzazione di Eulero + Linearizzazione TV (finite diff centrali)
def numerical_jacobian(f, x, u, p, eps_x=1e-5, eps_u=1e-5):
    """
    Jacobiani numerici centrali di f(x,u): Jx ~ df/dx, Ju ~ df/du
    f: (x,u,p) -> R^n
    """
    n = x.size
    m = u.size
    fx = f(x, u, p)
    Jx = np.zeros((n, n))
    Ju = np.zeros((n, m))

    # Derivate rispetto a x
    for i in range(n):
        dx = np.zeros(n); dx[i] = eps_x
        f_plus  = f(x + dx, u, p)
        f_minus = f(x - dx, u, p)
        Jx[:, i] = (f_plus - f_minus) / (2.0 * eps_x)

    # Derivate rispetto a u
    for j in range(m):
        du = np.zeros(m); du[j] = eps_u
        f_plus  = f(x, u + du, p)
        f_minus = f(x, u - du, p)
        Ju[:, j] = (f_plus - f_minus) / (2.0 * eps_u)

    return Jx, Ju, fx

def linearize_discretize(x_bar, u_bar, Ts, p):
    """
    A_k ≈ I + Ts * (∂f/∂x),  B_k ≈ Ts * (∂f/∂u),  g_k = x̄ + Ts f - A_k x̄ - B_k ū
    """
    Jx, Ju, fval = numerical_jacobian(f_cont, x_bar, u_bar, p)
    Ad = np.eye(x_bar.size) + Ts * Jx
    Bd = Ts * Ju
    g  = x_bar + Ts * fval - Ad @ x_bar - Bd @ u_bar
    return Ad, Bd, g

# Costruzione errore laterale e heading 
def lateral_error(X, Y, Xref, Yref, phiref):
    """
    e_c = sin(phi_ref)*(X - Xref) - cos(phi_ref)*(Y - Yref)
    """
    s = np.sin(phiref); c = np.cos(phiref)
    return s * (X - Xref) - c * (Y - Yref)

# MPC STEP (TV-QP)
def mpc_step(
    x0,
    u_prev,
    path_ref,      # array shape (N+1, 3): [X*, Y*, phi*] per k=0..N
    Ts=0.02,
    N=20,
    params=None,
    # Pesi costo
    q_c=6.0,       # peso errore laterale
    q_phi=0.5,     # peso heading vs phi*
    q_vx=0.5,      # peso tracking vx 
    R=np.diag([0.02, 2.0]),        # input
    Rd=np.diag([0.01, 5.0]),       # delta-u
    vref=None,     # float o array shape (N+1,)
    # Vincoli su input e rate
    u_bounds=(( -1.0,  1.0),   # d
              ( -0.6,  0.6)),  # delta [rad]
    du_bounds=((-0.5, 0.5),    # Δd per step
               (-0.3, 0.3)),   # Δdelta per step
    #limiti su stati (facoltativi)
    x_lo=None,
    x_hi=None,
    solver=cp.OSQP,
    verbose=False
):
   

    p = dict(Params)
    if params is not None:
        p.update(params)

    x0   = np.asarray(x0).reshape(6)
    u_pr = np.asarray(u_prev).reshape(2)
    path_ref = np.asarray(path_ref)
    assert path_ref.shape[0] == N + 1 and path_ref.shape[1] == 3 # shape corretta 

    # Riferimenti
    Xr = path_ref[:, 0]
    Yr = path_ref[:, 1]
    Pr = path_ref[:, 2]  # phi_ref

    if vref is None:
        vref = np.full(N + 1, x0[3])  # mantiene vx iniziale
    elif np.isscalar(vref):
        vref = np.full(N + 1, float(vref))
    else:
        vref = np.asarray(vref).reshape(N + 1)

    # 1) Traiettoria di riferimento nominale (x̄, ū) con Eulero
    xbar = np.zeros((6, N + 1))
    ubar = np.zeros((2, N))
    xbar[:, 0] = x0
    for k in range(N):
        ubar[:, k] = u_pr
        xbar[:, k + 1] = xbar[:, k] + Ts * f_cont(xbar[:, k], ubar[:, k], p)

    # 2) Precalcolo Ad, Bd, g per k=0..N-1
    Ad_list, Bd_list, g_list = [], [], []
    for k in range(N):
        Ad, Bd, g = linearize_discretize(xbar[:, k], ubar[:, k], Ts, p)
        Ad_list.append(Ad); Bd_list.append(Bd); g_list.append(g)

    # 3) Variabili QP
    X = cp.Variable((6, N + 1))
    U = cp.Variable((2, N))

    constraints = []
    constraints += [X[:, 0] == x0]

    # Dinamica (vincoli di uguaglianza)
    for k in range(N):
        Ad, Bd, g = Ad_list[k], Bd_list[k], g_list[k]
        constraints += [X[:, k + 1] == Ad @ X[:, k] + Bd @ U[:, k] + g]

    # Bounds input + rate
    (d_min, d_max), (del_min, del_max) = u_bounds
    (dd_min, dd_max), (ddel_min, ddel_max) = du_bounds

    for k in range(N):
        constraints += [
            U[0, k] >= d_min,   U[0, k] <= d_max,
            U[1, k] >= del_min, U[1, k] <= del_max
        ]
        if k == 0:
            constraints += [
                U[0, k] - u_pr[0] >= dd_min,   U[0, k] - u_pr[0] <= dd_max,
                U[1, k] - u_pr[1] >= ddel_min, U[1, k] - u_pr[1] <= ddel_max
            ]
        else:
            constraints += [
                U[0, k] - U[0, k-1] >= dd_min,   U[0, k] - U[0, k-1] <= dd_max,
                U[1, k] - U[1, k-1] >= ddel_min, U[1, k] - U[1, k-1] <= ddel_max
            ]

    # Bounds sugli stati (6 stati)
    if x_lo is not None:
        x_lo = np.asarray(x_lo).reshape(6)
        for k in range(N + 1):
            constraints += [X[:, k] >= x_lo]
    if x_hi is not None:
        x_hi = np.asarray(x_hi).reshape(6)
        for k in range(N + 1):
            constraints += [X[:, k] <= x_hi]

    # 4) Costo
    obj = 0
    # componenti: errore laterale ec, heading, vx, input e delta-input
    for k in range(N):
        # errore laterale (lineare in X,Y perché sin/cos(phi_ref) sono costanti al passo k)
        ec_k = lateral_error(X[0, k], X[1, k], Xr[k], Yr[k], Pr[k])
        obj += q_c * cp.square(ec_k)

        # heading vs riferimento
        obj += q_phi * cp.square(X[2, k] - Pr[k])

        # tracking vx
        obj += q_vx * cp.square(X[3, k] - vref[k])

        # costo input
        obj += cp.quad_form(U[:, k], R)

        # costo su delta-u
        if k == 0:
            du = U[:, k] - u_pr
        else:
            du = U[:, k] - U[:, k-1]
        obj += cp.quad_form(du, Rd)

    # terminal costs
    ec_T = lateral_error(X[0, N], X[1, N], Xr[N], Yr[N], Pr[N])
    obj += q_c * cp.square(ec_T)
    obj += q_phi * cp.square(X[2, N] - Pr[N])
    obj += q_vx * cp.square(X[3, N] - vref[N])

    prob = cp.Problem(cp.Minimize(obj), constraints)

    try:
        _ = prob.solve(solver=solver, warm_start=True, verbose=verbose)
        status = prob.status
    except Exception as e:
        return u_pr, f"error: {type(e).__name__}: {e}", {}

    if status not in ("optimal", "optimal_inaccurate"):
        return u_pr, status, {}

    u_cmd = np.array([U[0, 0].value, U[1, 0].value]).reshape(2)
    info = {
        "status": status,
        "objective": prob.value,
        "X": X.value,
        "U": U.value,
        "path_ref": path_ref,
        "vref": vref
    }
    return u_cmd, status, info


