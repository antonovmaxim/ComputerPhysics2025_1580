import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
np.random.seed(42)

# === Безразмерные параметры ===
EPSILON = 1.0
SIGMA = 1.0
MASS = 1.0
K_B = 1.0

# === Частица ===
class Particle:
    def __init__(self, pclsys, cord):
        self.pclsys = pclsys
        self.cord = cord.copy()
        self.vel = np.random.randn(2)

    def calc_force(self):
        F = np.zeros(2)
        for p in self.pclsys.particles:
            if p is self: continue
            r = self.cord - p.cord
            r -= self.pclsys.L * np.round(r / self.pclsys.L)
            r_abs = np.linalg.norm(r)
            if r_abs == 0: continue
            f_scalar = (24 * EPSILON / r_abs**2) * ((2 * (SIGMA / r_abs)**12) - (SIGMA / r_abs)**6)
            F += f_scalar * r
        return F

    def calc_Ek(self):
        return 0.5 * MASS * np.dot(self.vel, self.vel)

    def calc_Ep(self):
        ep = 0.0
        for p in self.pclsys.particles:
            if p is self: continue
            r = self.cord - p.cord
            r -= self.pclsys.L * np.round(r / self.pclsys.L)
            r_abs = np.linalg.norm(r)
            if r_abs == 0: continue
            ep += 4 * EPSILON * ((SIGMA / r_abs)**12 - (SIGMA / r_abs)**6)
        return ep

# === Система частиц ===
class ParticleSystem:
    def __init__(self, N, L):
        self.N = N
        self.L = L
        self.particles = []

        n_side = int(np.sqrt(N))
        a = L / n_side
        for i in range(n_side):
            for j in range(n_side):
                if len(self.particles) >= N: break
                pos = np.array([(i + 0.5) * a, (j + 0.5) * a])
                self.particles.append(Particle(self, pos))

        v_cm = sum(p.vel for p in self.particles) / N
        for p in self.particles:
            p.vel -= v_cm

    def update(self, dt):
        acc_old = [p.calc_force() / MASS for p in self.particles]
        for i, p in enumerate(self.particles):
            p.cord += p.vel * dt + 0.5 * acc_old[i] * dt**2
            p.cord %= self.L
        acc_new = [p.calc_force() / MASS for p in self.particles]
        for i, p in enumerate(self.particles):
            p.vel += 0.5 * (acc_old[i] + acc_new[i]) * dt

    def calc_Ek(self):
        return sum(p.calc_Ek() for p in self.particles)

    def calc_Ep(self):
        return sum(p.calc_Ep() for p in self.particles) / 2

    def temperature(self):
        return 2 * self.calc_Ek() / self.N

    def rescale_velocities(self, target_T):
        current_T = self.temperature()
        scale = np.sqrt(target_T / current_T)
        for p in self.particles:
            p.vel *= scale

    def calc_pressure(self):
        virial_sum = 0.0
        for p in self.particles:
            F = p.calc_force()
            virial_sum += np.dot(p.cord, F)
        V = self.L ** 2
        return (self.N * K_B * self.temperature() + 0.5 * virial_sum) / V

# === Расчёт g(r) ===
def compute_gr(particles, L, dr=0.1, r_max=None, bins=None):
    N = len(particles)
    coords = np.array([p.cord for p in particles])
    if r_max is None:
        r_max = L / 2
    if bins is None:
        bins = int(r_max / dr)

    g = np.zeros(bins)
    counts = np.zeros(bins)
    rho = N / (L ** 2)

    for i in range(N):
        for j in range(i + 1, N):
            rij = coords[i] - coords[j]
            rij -= L * np.round(rij / L)
            r = np.linalg.norm(rij)
            if r < r_max:
                bin_idx = int(r / dr)
                counts[bin_idx] += 2

    r_vals = dr * (np.arange(bins) + 0.5)
    shell_areas = 2 * np.pi * r_vals * dr
    norm = rho * N
    g = counts / (norm * shell_areas)
    return r_vals, g

# === Сбор статистики ===
class ExtendedStatsCollector:
    def __init__(self):
        self.ek_list = []
        self.ep_list = []
        self.temp_list = []
        self.pressure_list = []
        self.coord_snapshots = []

    def record(self, system):
        Ek = system.calc_Ek()
        Ep = system.calc_Ep()
        T = 2 * Ek / system.N
        P = system.calc_pressure()

        self.ek_list.append(Ek)
        self.ep_list.append(Ep)
        self.temp_list.append(T)
        self.pressure_list.append(P)
        self.coord_snapshots.append([p.cord.copy() for p in system.particles])

    def summarize(self):
        N = len(self.ek_list)
        return {
            "mean_E": np.mean(np.array(self.ek_list) + np.array(self.ep_list)),
            "std_E": np.std(np.array(self.ek_list) + np.array(self.ep_list)) / np.sqrt(N),
            "mean_T": np.mean(self.temp_list),
            "std_T": np.std(self.temp_list) / np.sqrt(N),
            "mean_P": np.mean(self.pressure_list),
            "std_P": np.std(self.pressure_list) / np.sqrt(N),
        }

# === Симуляция с возвратом g(r) ===
def simulate_with_gr_return_gr(N, L, T0=0.45, dt=0.005, n_therm_steps=50, n_steps=200, dr=0.1):
    system = ParticleSystem(N, L)
    stats = ExtendedStatsCollector()

    for _ in tqdm(range(n_therm_steps), desc = "Термализация"):
        system.update(dt)
        system.rescale_velocities(T0)

    for _ in tqdm(range(n_steps), desc = "Шаги"):
        system.update(dt)
        stats.record(system)

    r_vals, g_vals = compute_gr(system.particles, L, dr=dr)
    return stats.summarize(), r_vals, g_vals

# === Главная функция ===
def main():
    cases = [
        ("Твёрдое тело", 0.1, 'white'),
        ("Жидкость", 8, 'blue'),
        ("Газ", 25, 'red')
    ]

    N = 100
    L = 10.0

    fig = go.Figure()

    for name, T0, c in cases:
        print(f"--- Диагностика фазы: {name} (T0 = {T0}) ---")
        summary, r_vals, g_vals = simulate_with_gr_return_gr(N=N, L=L, T0=T0)
        for key, val in summary.items():
            print(f"{key}: {val:.4f}         ")
        fig.add_trace(go.Scatter(x=r_vals, y=g_vals, mode='lines', name=name, line=dict(color=c)))

    fig.update_layout(
        title_text="Радиальная функция распределения g(r) для разных фаз",
        xaxis_title="r",
        yaxis_title="g(r)",
        template="plotly_dark",
        legend_title="Фаза"
    )
    fig.show()

if __name__ == '__main__':
    main()