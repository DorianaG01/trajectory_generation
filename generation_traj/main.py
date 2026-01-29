import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from mpc_6stati import mpc_step, f_cont, Params  

# Load vehicle parameters
p = Params

def d_steady_state(v):
    """
    Calculates the steady-state motor duty cycle 'd' to maintain a constant velocity 'v'.
    Balances motor thrust against rolling resistance and aerodynamic drag.
    """
    # Sum of resistances (rolling + aerodynamic)
    num = p["Cr0"] + p["Cr2"] * v**2 
    # Available motor thrust which decreases as speed increases
    den = p["Cm1"] - p["Cm2"] * v 
    return num / den

# Initial target velocity and corresponding steady-state control
v0_target = 1.0                        
u_prev = np.array([d_steady_state(v0_target), 0.0])
print(f"Initial State: v0_target={v0_target:.2f} m/s, u_prev={u_prev}")

# --- REFERENCE VELOCITY PROFILES ---
# Each function builds a velocity array of length N+1 for the prediction horizon.

def vref_profile_ramp_cruise(N, Ts, v0=0.8, v_cruise=2.0, tramp=2.0):
    """Velocity accelerates linearly from v0 to v_cruise over tramp seconds."""
    t = np.arange(N + 1) * Ts
    v = v0 + (v_cruise - v0) * np.clip(t / tramp, 0.0, 1.0)
    return v

def vref_profile_trapezoid(N, Ts, v0=0.8, vmax=2.0, t_acc=2.0, t_flat=3.0, t_dec=2.0):
    """Trapezoidal velocity profile: acceleration, cruise, then deceleration."""
    t = np.arange(N + 1) * Ts
    v = np.full(N + 1, v0)
    v = np.where(t <= t_acc, v0 + (vmax - v0) * (t / t_acc), vmax)
    v = np.where(t > t_acc + t_flat,
                 vmax - (vmax - v0) * ((t - (t_acc + t_flat)) / t_dec),
                 v)
    return np.clip(v, v0, vmax)

def vref_profile_sine(N, Ts, v_mean=1.5, v_amp=0.5, period_s=6.0):
    """Velocity oscillates sinusoidally around a mean value."""
    t = np.arange(N + 1) * Ts
    return v_mean + v_amp * np.sin(2 * np.pi * t / period_s)

# --- PATH GENERATION ---

def ref_window_from_x_with_vref(x_start, N, Ts, vref_seq):
    """
    Generates the reference trajectory (X*, Y*, phi*) starting from x_start.
    Integrates vref_seq to find X positions and maps them to a geometric path.
    """
    vref_seq = np.asarray(vref_seq).reshape(N + 1)
    xs = np.zeros(N + 1)
    xs[0] = x_start
    for k in range(N):
        # Calculate X-axis advancement based on reference velocity
        xs[k + 1] = xs[k] + vref_seq[k] * Ts  
    
    # Geometric path: Parabola (e.g., y = 0.1 * x^2)
    ys = 0.1 * xs**2
    dydx = 0.2 * xs # Derivative for heading calculation
    phs = np.arctan(dydx) # Desired heading (phi*)
    
    return np.stack([xs, ys, phs], axis=1)

# --- MPC SIMULATION SETUP ---

Ts = 0.02       # Simulation sampling time
sim_steps = 600 # Total simulation duration steps
N = 40           # MPC prediction horizon (number of steps)

# Initial state: [X, Y, phi, vx, vy, omega]
x = np.array([0.0, 0.5, 0.0, v0_target, 0.0, 0.0])  

traj_X = []     # Store state history
traj_U = []     # Store control history
path_refs = []  # Store reference windows for animation

# --- SIMULATION LOOP ---

for t in range(sim_steps):
    # Select reference velocity profile
    vref_seq = vref_profile_ramp_cruise(N, Ts, v0=0.8, v_cruise=2.0, tramp=2.0)
    
    # Generate the reference path window starting from the current vehicle X-position
    path_ref = ref_window_from_x_with_vref(x_start=x[0], N=N, Ts=Ts, vref_seq=vref_seq)
    path_refs.append(path_ref)

    # Compute MPC control action
    u_cmd, status, info = mpc_step(x, u_prev, path_ref, Ts=Ts, N=N, params=Params, vref=vref_seq)
    
    # Apply dynamics (Euler integration)
    x = x + Ts * f_cont(x, u_cmd, Params) 

    traj_X.append(x.copy())
    traj_U.append(u_cmd.copy())
    u_prev = u_cmd

# Convert lists to arrays for analysis
traj_X = np.array(traj_X)
traj_U = np.array(traj_U)

# --- STATISTICAL ANALYSIS ---

t_u = np.arange(traj_U.shape[0]) * Ts
d_seq = traj_U[:, 0]        # Accelerator sequence
delta_seq = traj_U[:, 1]    # Steering sequence

print(f"\nAccelerator 'd_seq' Statistics:")
print(f"  - Mean: {np.mean(d_seq):.4f}")
print(f"  - Std Dev: {np.std(d_seq):.4f}")
print(f"  - Range: [{np.min(d_seq):.4f}, {np.max(d_seq):.4f}]")

print(f"\nSteering 'delta_seq' Statistics:")
print(f"  - Mean: {np.mean(delta_seq):.4f}")
print(f"  - Std Dev: {np.std(delta_seq):.4f}")
print(f"  - Range: [{np.min(delta_seq):.4f}, {np.max(delta_seq):.4f}]")

# --- PLOTTING ---

fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig.suptitle("MPC Control Inputs and Statistical Means")

# Plot Accelerator (d)
axs[0].plot(t_u, d_seq, linewidth=1.8, label='Duty Cycle (d)')
axs[0].axhline(np.mean(d_seq), color='r', linestyle='--', label=f'Mean: {np.mean(d_seq):.2f}')
axs[0].set_ylabel("d [-]")
axs[0].grid(True, linestyle=":")
axs[0].legend()

# Plot Steering (delta)
axs[1].plot(t_u, delta_seq, linewidth=1.8, label='Steering (delta)')
axs[1].axhline(np.mean(delta_seq), color='r', linestyle='--', label=f'Mean: {np.mean(delta_seq):.2f}')
axs[1].set_ylabel("δ [rad]")
axs[1].set_xlabel("Time [s]")
axs[1].grid(True, linestyle=":")
axs[1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- ANIMATION: VEHICLE VS REFERENCE ---



T = len(traj_X)
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(np.min(traj_X[:,0]) - 1, np.max(traj_X[:,0]) + 1)
ax.set_ylim(np.min(traj_X[:,1]) - 1, np.max(traj_X[:,1]) + 1)
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_title("MPC Tracking: Vehicle vs Reference")

# Visual elements
trail, = ax.plot([], [], 'b-', linewidth=2, label="Vehicle path")
pt = ax.scatter([], [], s=40, c='r', label="Current Position")
arrow = ax.quiver([], [], [], [], angles='xy', scale_units='xy', scale=1, color='r')
ref_ln, = ax.plot([], [], 'k--', linewidth=1.5, label="Reference Window")

ax.legend(loc="best")

def init():
    trail.set_data([], [])
    pt.set_offsets(np.empty((0, 2)))
    ref_ln.set_data([], [])
    return trail, pt, ref_ln, arrow

def update(i):
    # Path traveled up to step i
    trail.set_data(traj_X[:i+1, 0], traj_X[:i+1, 1])
    
    # Vehicle current position
    pt.set_offsets(np.array([[traj_X[i, 0], traj_X[i, 1]]]))

    # Current heading arrow
    phi = traj_X[i, 2]
    dx, dy = np.cos(phi), np.sin(phi)
    arrow.set_offsets([traj_X[i, 0], traj_X[i, 1]])
    arrow.set_UVC(dx, dy)

    # Current MPC reference window
    pref = path_refs[min(i, len(path_refs)-1)]
    ref_ln.set_data(pref[:, 0], pref[:, 1])

    return trail, pt, ref_ln, arrow

ani = animation.FuncAnimation(
    fig, update, frames=T, init_func=init,
    blit=False, interval=40  
)

# Save result
ani.save("mpc_tracking.gif", writer=animation.PillowWriter(fps=25))
print("Animation saved as: mpc_tracking.gif")
plt.show()
