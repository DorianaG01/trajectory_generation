import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

Params = {
    "Cm1": 0.287, "Cm2": 0.0545, "Cr0": 0.0518, "Cr2": 0.00035,
    "Br": 3.3852, "Cr": 1.2691, "Dr": 0.1737, "Bf": 2.579,
    "Cf": 1.2,    "Df": 0.192, "m": 0.041, "Iz": 27.8e-6,
    "lf": 0.029, "lr": 0.033, "g": 9.81, "maxAlpha": 0.6, "vx_zero": 0.3,
}
np.random.seed(42)

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def tire_forces(x, u, p):
    _, _, _, vx, vy, omega = x; d, delta = u
    vx_eff = max(abs(vx), p["vx_zero"])
    alpha_f = -np.arctan2(omega * p["lf"] + vy, vx_eff) + delta
    alpha_r =  np.arctan2(omega * p["lr"] - vy, vx_eff)
    alpha_f = clamp(alpha_f, -p["maxAlpha"], p["maxAlpha"])
    Fy_f = p["Df"] * np.sin(p["Cf"] * np.arctan(p["Bf"] * alpha_f))
    Fy_r = p["Dr"] * np.sin(p["Cr"] * np.arctan(p["Br"] * alpha_r))
    Frx = (p["Cm1"] - p["Cm2"] * vx_eff) * d - p["Cr0"] - p["Cr2"] * (vx_eff**2)
    return Fy_f, Fy_r, Frx

def f_cont(x, u, p):
    X, Y, phi, vx, vy, omega = x; d, delta = u
    m, Iz, lf, lr = p["m"], p["Iz"], p["lf"], p["lr"]
    Fy_f, Fy_r, Frx = tire_forces(x, u, p)
    Xdot = vx*np.cos(phi) - vy*np.sin(phi)
    Ydot = vx*np.sin(phi) + vy*np.cos(phi)
    phidot = omega
    vxdot = (Frx - Fy_f*np.sin(delta) + m*vy*omega) / m
    vydot = (Fy_r + Fy_f*np.cos(delta) - m*vx*omega) / m
    omegadot = (Fy_f*lf*np.cos(delta) - Fy_r*lr) / Iz
    return np.array([Xdot, Ydot, phidot, vxdot, vydot, omegadot])

def simulate_trajectory(x0, U_sim, time_step, p):
    sim_steps = len(U_sim)
    history_X = np.empty((sim_steps + 1, 6))
    history_X[0, :] = x0
    for k in range(sim_steps):
        x_dot = f_cont(history_X[k, :], U_sim[k, :], p)
        history_X[k + 1, :] = history_X[k, :] + time_step * x_dot
    return history_X

def create_spline_signal(num_steps, Ts, d_mean, d_std, delta_mean, delta_std):

    if num_steps <= 1: return np.array([d_mean]), np.array([delta_mean])

    checkpoint_interval_s = np.random.uniform(3.0, 5.0)
    checkpoint_interval_steps = max(1, int(round(checkpoint_interval_s / Ts)))
    idx = np.arange(0, num_steps, checkpoint_interval_steps)

    if idx[-1] != num_steps - 1: idx = np.append(idx, num_steps - 1)
    d_chk, delta_chk = np.random.normal(d_mean, d_std, len(idx)), np.random.normal(delta_mean, delta_std, len(idx))
    s_d, s_delta = CubicSpline(idx, d_chk, bc_type='natural'), CubicSpline(idx, delta_chk, bc_type='natural')
    t = np.arange(num_steps)

    return s_d(t), s_delta(t)

def generate_smooth_profiles(total_steps, Ts, stats, mode='random'):

    if mode == 'random': mode = np.random.choice(['straight', 'sinusoid'], p=[0.5, 0.5])

    transient_duration_s = np.random.uniform(1.5, 3.0)
    transient_steps = min(int(transient_duration_s / Ts), total_steps)
    steady_state_steps = total_steps - transient_steps

    d_tr, delta_tr = create_spline_signal(num_steps=transient_steps, Ts=Ts, d_mean=stats['d_mean'], d_std=stats['d_std']*0.2, delta_mean=stats['delta_mean'], delta_std=stats['delta_std']*0.3)
    
    if steady_state_steps > 0:
        d_st = np.random.normal(stats['d_mean'], stats['d_std']*0.05, steady_state_steps)
        if mode == 'sinusoid':
            t_st = np.arange(steady_state_steps) * Ts
            period, amplitude = np.random.uniform(4.0, 8.0), np.random.uniform(stats['delta_std']*1.5, stats['delta_std']*2.5)
            phase, omega = np.random.uniform(0, 2*np.pi), 2*np.pi/period
            sinusoid, noise = amplitude*np.sin(omega*t_st + phase), np.random.normal(0, stats['delta_std']*0.1, steady_state_steps)
            delta_st = stats['delta_mean'] + sinusoid + noise
        else: delta_st = np.random.normal(stats['delta_mean'], stats['delta_std']*0.01, steady_state_steps)
        d_profile, delta_profile = np.concatenate([d_tr, d_st]), np.concatenate([delta_tr, delta_st])
    else: d_profile, delta_profile = d_tr, delta_tr

    return d_profile, delta_profile, mode

def apply_du_bounds(u, du_min, du_max):
    out = np.empty_like(u); out[0] = u[0]
    for k in range(1, len(u)):
        du = np.clip(u[k] - out[k-1], du_min, du_max)
        out[k] = out[k-1] + du
    return out

def plot_sample_controls(dataset, trajectory_id):
    traj_data = dataset[dataset['trajectory_id'] == trajectory_id]

    clean_traj = traj_data[traj_data['noise_type'] == 'clean'].copy().dropna(subset=['d', 'delta'])
    noisy_traj = traj_data[traj_data['noise_type'] == 'noisy'].copy().dropna(subset=['d', 'delta'])

    if clean_traj.empty or noisy_traj.empty:
        print(f"Dati mancanti per la traiettoria {trajectory_id}.")
        return
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    mode = clean_traj['mode'].iloc[0]
    fig.suptitle(f'Confronto Controlli - Traiettoria {trajectory_id} (Modalità: {mode})', fontsize=16)
    axs[0].plot(clean_traj['t'], clean_traj['d'], lw=2, color='royalblue', label='Clean (Spline)')
    axs[0].plot(noisy_traj['t'], noisy_traj['d'], lw=1, color='firebrick', alpha=0.8, label='Noisy + Limited')
    axs[0].set_ylabel('d [-]'); axs[0].grid(True, linestyle=':'); axs[0].legend()
    axs[1].plot(clean_traj['t'], clean_traj['delta'], lw=2, color='royalblue', label='Clean (Spline)')
    axs[1].plot(noisy_traj['t'], noisy_traj['delta'], lw=1, color='firebrick', alpha=0.8, label='Noisy + Limited')
    axs[1].set_ylabel('delta [rad]'); axs[1].set_xlabel('Tempo [s]'); axs[1].grid(True, linestyle=':'); axs[1].legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()
    
def animate_trajectories(dataset, num_to_animate, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal', adjustable='box')
    title = f'Animazione di {num_to_animate} Traiettorie'
    if save_path: title += f" ({'Clean' if 'clean' in save_path else 'Noisy'})"
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Posizione X [m]'); ax.set_ylabel('Posizione Y [m]'); ax.grid(True, linestyle=':')
    trajectories_data = []
    for i in range(min(num_to_animate, int(dataset['trajectory_id'].max()) + 1)):
        traj = dataset[dataset['trajectory_id'] == i]
        trajectories_data.append({'x': traj['X'].values, 'y': traj['Y'].values, 't': traj['t'].values})
    x_min, x_max = min(d['x'].min() for d in trajectories_data)-1, max(d['x'].max() for d in trajectories_data)+1
    y_min, y_max = min(d['y'].min() for d in trajectories_data)-1, max(d['y'].max() for d in trajectories_data)+1
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    lines = [ax.plot([], [], lw=2, label=f'Traiettoria {i}')[0] for i in range(num_to_animate)]
    points = [ax.plot([], [], 'o', markersize=8, color=line.get_color())[0] for line in lines]
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.legend(loc='best')

    def init():
        for line, point in zip(lines, points): line.set_data([], []); point.set_data([], [])
        time_text.set_text('')
        return lines + points + [time_text]
    
    def animate(i):
        for j, data in enumerate(trajectories_data):
            lines[j].set_data(data['x'][:i+1], data['y'][:i+1])
            points[j].set_data([data['x'][i]], [data['y'][i]])
        time_text.set_text(f'Tempo = {trajectories_data[0]["t"][i]:.2f} s')
        return lines + points + [time_text]
    ani = FuncAnimation(fig, animate, frames=len(trajectories_data[0]['t']), init_func=init, interval=30, blit=True)
    if save_path:
        print(f"\nSalvataggio animazione in corso su '{save_path}'...")
        writer = PillowWriter(fps=int(1000/30)); ani.save(save_path, writer=writer, dpi=120)
        print("Salvataggio completato.")
    plt.show()

if __name__ == "__main__":
    num_traj = 5000
    traj_duration = 12.0
    time_step = 0.02
    
    #output_csv_clean = "vehicle_mpc_clean.csv"
    output_csv_noisy = "vehicle_mpc_noisy.csv"
    #output_animation_noisy = "animation_noisy.gif"

    mpc_stats = {'d_mean': 0.2161, 'd_std': 0.1314, 'delta_mean': 0.0035, 'delta_std': 0.0338}
    du_bounds = ((-0.1, 0.1), (-0.04, 0.04))

    all_trajectories_data = []
    sim_steps_per_traj = int(traj_duration / time_step)
    
    for i in tqdm(range(num_traj), desc="Generando Traiettorie"):

        v_init, phi_init, beta_init = (0.4, 1.5), (-np.pi, np.pi), (-0.2, 0.2)
        omega_init= np.random.uniform(-1, 1)
        v, phi, beta = np.random.uniform(*v_init), np.random.uniform(*phi_init), np.random.uniform(*beta_init)
        vx, vy = v*np.cos(beta), v*np.sin(beta)
        x, y = np.random.uniform(-2, 2), np.random.uniform(-2, 2)
        

        x0 = np.array([x, y, phi, vx, vy, omega_init])

        d_clean, delta_clean, mode = generate_smooth_profiles(sim_steps_per_traj, time_step, mpc_stats)
        d_clean_limited = np.clip(apply_du_bounds(d_clean, *du_bounds[0]), -1.0, 1.0)
        delta_clean_limited = np.clip(apply_du_bounds(delta_clean, *du_bounds[1]), -0.6, 0.6)
        U_clean = np.stack([d_clean_limited, delta_clean_limited], axis=1)

        noise_d = np.random.normal(0, mpc_stats['d_std'] * 0.1, sim_steps_per_traj)
        noise_delta = np.random.normal(0, mpc_stats['delta_std'] * 0.1, sim_steps_per_traj)
        d_noisy = np.clip(apply_du_bounds(d_clean + noise_d, *du_bounds[0]), -1.0, 1.0)
        delta_noisy = np.clip(apply_du_bounds(delta_clean + noise_delta, *du_bounds[1]), -0.6, 0.6)
        U_noisy = np.stack([d_noisy, delta_noisy], axis=1)

        history_X_clean = simulate_trajectory(x0, U_clean, time_step, Params)
        history_X_noisy = simulate_trajectory(x0, U_noisy, time_step, Params)

        for noise_type, history_X, U_sim in [('clean', history_X_clean, U_clean), ('noisy', history_X_noisy, U_noisy)]:
            traj_df = pd.DataFrame({
                't': np.arange(sim_steps_per_traj + 1) * time_step, 'X': history_X[:, 0], 'Y': history_X[:, 1], 
                 'vx': history_X[:, 3], 'vy': history_X[:, 4], 'omega': history_X[:, 5],
                'd': np.append(U_sim[:, 0], np.nan), 'delta': np.append(U_sim[:, 1], np.nan),
                'trajectory_id': i, 'noise_type': noise_type, 'mode': mode
            })
            all_trajectories_data.append(traj_df)
    
    final_dataset = pd.concat(all_trajectories_data, ignore_index=True)
    
    #clean_dataset = final_dataset[final_dataset['noise_type'] == 'clean'].copy()
    noisy_dataset = final_dataset[final_dataset['noise_type'] == 'noisy'].copy()
    
    #if 'noise_type' and 'mode' in clean_dataset.columns:
        #clean_dataset = clean_dataset.drop(columns=['noise_type', 'mode'])
    if 'noise_type' and 'mode' in noisy_dataset.columns:
        noisy_dataset = noisy_dataset.drop(columns=['noise_type', 'mode'])
        
    #clean_dataset.to_csv(output_csv_clean, index=False)
    #print(f"   -> Dataset pulito salvato come '{output_csv_clean}'")
    noisy_dataset.to_csv(output_csv_noisy, index=False)
    print(f"   -> Dataset rumoroso salvato come '{output_csv_noisy}'")

    if not final_dataset.empty:
        sinusoid_example_id = final_dataset[final_dataset['mode'] == 'sinusoid']['trajectory_id'].iloc[0]
        plot_sample_controls(final_dataset, sinusoid_example_id)
        
        #animate_trajectories(noisy_dataset, num_to_animate=10, save_path=output_animation_noisy)