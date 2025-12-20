import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, List
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# -----------------------------
#  PARAMETRI DEL MODELLO
# -----------------------------
Params = {
    "Cm1": 0.287, "Cm2": 0.0545, "Cr0": 0.0518, "Cr2": 0.00035,
    "Br": 3.3852, "Cr": 1.2691, "Dr": 0.1737, "Bf": 2.579,
    "Cf": 1.2,    "Df": 0.192, "m": 0.041, "Iz": 27.8e-6,
    "lf": 0.029, "lr": 0.033, "g": 9.81, "maxAlpha": 0.6, "vx_zero": 0.3,
}

@dataclass
class ControlRules:
    """Regole morbide con rumore contenuto (stabilità yaw)."""
    v_turn_max: float = 1.2
    v_high: float = 4.0
    d_range: Tuple[float, float] = (0.0, 0.33)
    delta_turn_range: Tuple[float, float] = (0.015, 0.04)
    delta_straight_noise: float = 0.004
    d_noise_std: float = 0.008          
    delta_noise_std: float = 0.002      
    meas_noise_std: Dict[str, float] = None
    def __post_init__(self):
        if self.meas_noise_std is None:
            self.meas_noise_std = {
    "X":    0.05,   # 5 mm
    "Y":    0.05,   # 5 mm
    "phi":  0.003,   # ≈ 0.17° 
    "vx":   0.010,   # 1 cm/s
    "vy":   0.003,   # 3 mm/s
    "omega":0.030,   # ≈ 1.7°/s
}

# -----------------------------
#  MODELLO VEICOLO
# -----------------------------
def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def tire_forces(x, u, p):
    _, _, _, vx, vy, omega = x; d, delta = u
    vx_eff = max(abs(vx), p["vx_zero"])
    alpha_f = -np.arctan2(omega * p["lf"] + vy, vx_eff) + delta
    alpha_r =  np.arctan2(omega * p["lr"] - vy, vx_eff)
    alpha_f = clamp(alpha_f, -p["maxAlpha"], p["maxAlpha"])
    alpha_r = clamp(alpha_r, -p["maxAlpha"], p["maxAlpha"])
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

def euler_step(x, u, p, Ts):
    return x + Ts * f_cont(x, u, p)


def sample_controls_piecewise(sim_steps, Ts, x0, p, rules: ControlRules, seed=42):
    prev_delta_cmd, delta_rate_max = 0.0, 0.30
    rng = np.random.default_rng(seed)
    U_sim, modes = np.zeros((sim_steps, 2)), [""] * sim_steps
    x_shadow = np.array(x0, dtype=float)
    v_floor, d_boost_min = 0.35, 0.15
    i = 0
    prev_mode = "init"
    while i < sim_steps:
        if prev_mode in ("turn_left", "turn_right"):
            # Se abbiamo appena finito una curva, FORZA un segmento dritto
            mode = rng.choice(["accelerate", "cruise"], p=[0.5, 0.5])
        else:
            # Se eravamo dritti, possiamo scegliere qualsiasi cosa
            mode = rng.choice(["accelerate", "cruise", "turn_left", "turn_right"], 
                              p=[0.35, 0.35, 0.15, 0.15])
        seg_len = max(1, int(np.round(rng.uniform(0.4, 1.5) / Ts)))
        v = np.hypot(x_shadow[3], x_shadow[4])
        if mode == "accelerate":
            if v >= rules.v_high: mode = "cruise"
            d, delta = rng.uniform(0.3, rules.d_range[1]), rng.normal(0.0, rules.delta_straight_noise)
        elif mode == "cruise":
            d, delta = rng.uniform(-0.05, 0.2), rng.normal(0.0, rules.delta_straight_noise)
        elif mode in ("turn_left", "turn_right"):
            d = rng.uniform(0.0, 0.15) if v > rules.v_turn_max else rng.uniform(0.05, 0.25)
            mag = rng.uniform(*rules.delta_turn_range)
            scale = min(1.0, rules.v_turn_max / max(v, 1e-3))
            delta = (mag if mode == "turn_left" else -mag) * scale
        if v < 0.5:
            mode, d, delta = "accelerate", rng.uniform(0.5, 1.0), rng.normal(0.0, rules.delta_straight_noise)
            seg_len = max(seg_len, int(round(0.3 / Ts)))
        turning = (mode in ("turn_left", "turn_right"))
        if turning and v > rules.v_turn_max: d = min(d, 0.15)
        if turning and v < 0.25: d = max(d, 0.05)
        for _ in range(seg_len):
            if i >= sim_steps: break
            v = np.hypot(x_shadow[3], x_shadow[4])
            if v < v_floor: d = max(d, d_boost_min)
            d_k = float(np.clip(d, *rules.d_range))
            delta_k = float(np.clip(delta, -0.6, 0.6))
            max_step = delta_rate_max * Ts
            delta_k = float(np.clip(delta_k, prev_delta_cmd - max_step, prev_delta_cmd + max_step))
            prev_delta_cmd = delta_k
            if v < v_floor: d_k = max(d_k, d_boost_min)
            U_sim[i, 0], U_sim[i, 1], modes[i] = d_k, delta_k, mode
            x_shadow = euler_step(x_shadow, U_sim[i], p, Ts)
            x_shadow[3] = max(x_shadow[3], 0.0)
            x_shadow[5] = float(np.clip(x_shadow[5], -6, 6))
            i += 1
        prev_mode = mode
    return U_sim, modes


def generate_dataset(num_traj, T, Ts, seed=2025, smooth_states_for_plots=True):
    rules = ControlRules()
    rng = np.random.default_rng(seed)
    sim_steps_per_traj = int(np.round(T / Ts))
    all_trajectories_data: List[pd.DataFrame] = []

    print(f"Inizio generazione di {num_traj} traiettorie...")
    for i in tqdm(range(num_traj), desc="Generando Traiettorie"):
        x0 = np.array([
            rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-np.pi, np.pi),
            rng.uniform(0.2, 0.6), rng.uniform(-0.05, 0.05), rng.uniform(-1, 1)
        ], dtype=float)

        U_sim, modes = sample_controls_piecewise(sim_steps_per_traj, Ts, x0, Params, rules, seed=seed+i)

        # integrazione modello fisico
        history_X_truth = np.empty((sim_steps_per_traj + 1, 6))
        history_X_truth[0, :] = x0
        for k in range(sim_steps_per_traj):
            x_dot = f_cont(history_X_truth[k, :], U_sim[k, :], Params)
            history_X_truth[k+1, :] = history_X_truth[k, :] + Ts*x_dot
            history_X_truth[k+1, 3] = max(history_X_truth[k+1, 3], 0.0)
            history_X_truth[k+1, 5] = float(np.clip(history_X_truth[k+1, 5], -6, 6))

       
        # misure rumorose
        N = sim_steps_per_traj + 1
        noise_rng = np.random.default_rng(12345 + i)
        noise = np.column_stack([
            noise_rng.normal(0, rules.meas_noise_std["X"], N),
            noise_rng.normal(0, rules.meas_noise_std["Y"], N),
            noise_rng.normal(0, rules.meas_noise_std["phi"], N),
            noise_rng.normal(0, rules.meas_noise_std["vx"], N),
            noise_rng.normal(0, rules.meas_noise_std["vy"], N),
            noise_rng.normal(0, rules.meas_noise_std["omega"], N),
        ])
        history_X_meas = history_X_truth + noise

        t_steps = np.arange(N) * Ts
        modes_full = modes + [""]

        for noise_type, state_history in [('clean', history_X_truth), ('noisy', history_X_meas)]:
            data = {
                't': t_steps,
                'X': state_history[:, 0], 'Y': state_history[:, 1],'phi': state_history[:, 2],
                'vx': state_history[:, 3], 'vy': state_history[:, 4], 'omega': state_history[:, 5],
                'd': np.append(U_sim[:, 0], np.nan),
                'delta': np.append(U_sim[:, 1], np.nan),
                'trajectory_id': i, 'noise_type': noise_type, 'mode': modes_full
            }
            if noise_type == 'clean':
                data['phi'] = state_history[:, 2]
            traj_df = pd.DataFrame(data)
            all_trajectories_data.append(traj_df)

    combined_dataset = pd.concat(all_trajectories_data, ignore_index=True, sort=False)
    print("Generazione dataset completata.")
    return combined_dataset

def animate_trajectories(dataset, num_to_animate=10, interval=10, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    title = f"Animazione di {num_to_animate} Traiettorie"
    if save_path: title += f" ({'Clean' if 'clean' in save_path else 'Noisy'})"
    ax.set_title(title, fontsize=16)
    traj_ids = sorted(dataset['trajectory_id'].unique().tolist())
    if not traj_ids: raise ValueError("Dataset vuoto.")
    num_to_animate = min(num_to_animate, len(traj_ids))
    selected_ids = traj_ids[:num_to_animate]
    trajectories_data = [{'x': df['X'].values, 'y': df['Y'].values, 't': df['t'].values, 'id': tid}
                         for tid in selected_ids if not (df := dataset[dataset['trajectory_id'] == tid].sort_values('t')).empty]
    if not trajectories_data:
        print("Nessuna traiettoria trovata per l'animazione.")
        return
    x_min, x_max = min(d['x'].min() for d in trajectories_data), max(d['x'].max() for d in trajectories_data)
    y_min, y_max = min(d['y'].min() for d in trajectories_data), max(d['y'].max() for d in trajectories_data)
    pad_x, pad_y = 0.05 * max(1e-6, (x_max - x_min)), 0.05 * max(1e-6, (y_max - y_min))
    max_len = max(len(d['t']) for d in trajectories_data)
    ax.set_xlabel('Posizione X [m]', fontweight='bold'); ax.set_ylabel('Posizione Y [m]', fontweight='bold')
    ax.grid(True); ax.set_aspect('equal', adjustable='box'); ax.set_xlim(x_min - pad_x, x_max + pad_x); ax.set_ylim(y_min - pad_y, y_max + pad_y)
    lines = [ax.plot([], [], lw=2, label=f'Traj {d["id"]}')[0] for d in trajectories_data]
    points = [ax.plot([], [], 'o', markersize=6, color=line.get_color())[0] for line in lines]
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
    ax.legend(loc='best')
    def init():
        for line, point in zip(lines, points): line.set_data([], []); point.set_data([], [])
        time_text.set_text('')
        return lines + points + [time_text]
    def animate(i):
        for j, d in enumerate(trajectories_data):
            idx = min(i, len(d['t']) - 1)
            lines[j].set_data(d['x'][:idx+1], d['y'][:idx+1])
            points[j].set_data([d['x'][idx]], [d['y'][idx]])
        t_show = trajectories_data[0]['t'][min(i, len(trajectories_data[0]['t']) - 1)]
        time_text.set_text(f'Tempo = {t_show:.2f} s')
        return lines + points + [time_text]
    ani = FuncAnimation(fig, animate, frames=max_len, init_func=init, interval=interval, blit=True)
    if save_path:
        try:
            writer = PillowWriter(fps=max(1, int(1000/interval)))
            ani.save(save_path, writer=writer, dpi=150)
            print(f"Animazione salvata in: {save_path}")
        except Exception as e: print(f"Errore durante salvataggio animazione: {e}")
    plt.show()

if __name__ == "__main__":
    num_traj = 5000
    duration = 12.0
    time_step = 0.01

    output_csv_clean = "vehicle_piecewise_clean.csv"
    output_csv_noisy = "vehicle_piecewise_noisy.csv"
    output_gif_clean = "animation_piecewise_clean.gif"
    output_gif_noisy = "animation_piecewise_noisy.gif"

    # 1. Genera il dataset combinato
    combined_dataset = generate_dataset(
        num_traj=num_traj,
        T=duration,
        Ts=time_step,
        seed=42
    )

    # 2. Salva i dataset separati
    clean_dataset = combined_dataset[combined_dataset['noise_type'] == 'clean'].copy()
    noisy_dataset = combined_dataset[combined_dataset['noise_type'] == 'noisy'].copy()
    
    if 'noise_type' and 'mode' in clean_dataset.columns:
        clean_dataset = clean_dataset.drop(columns=['noise_type', 'mode'])
    if 'noise_type' and 'mode' in noisy_dataset.columns:
        noisy_dataset = noisy_dataset.drop(columns=['noise_type', 'mode', 'phi'])

    clean_dataset.to_csv(output_csv_clean, index=False)
    print(f"   -> Dataset pulito salvato come '{output_csv_clean}'")
    noisy_dataset.to_csv(output_csv_noisy, index=False)
    print(f"   -> Dataset rumoroso salvato come '{output_csv_noisy}'")
    
    # 3. Visualizzazione e Animazione
    if not combined_dataset.empty:

        animate_trajectories(clean_dataset, num_to_animate=10, save_path=output_gif_clean)
        animate_trajectories(noisy_dataset, num_to_animate=10, save_path=output_gif_noisy) 