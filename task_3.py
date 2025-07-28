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

def calc_pressure_kinetic(self):
    V = self.L ** 2
    total = 0.0
    for p in self.particles:
        a = p.calc_force() / MASS
        total += np.dot(p.vel, a)
    return MASS * total / V
def task3_pressure_vs_density():
    T0 = 0.45
    N = 100
    rhos = np.linspace(0.1, 1.0, 10)
    
    E_list = []
    P_virial_list = []
    P_kinetic_list = []

    print("--- ЗАДАЧА 3: E(ρ), P(ρ) при T = 0.45 ---")

    for rho in (rhos):
        V = N / rho
        L = np.sqrt(V)
        system = ParticleSystem(N, L)

        # Термализация
        for _ in tqdm(range(5)):
            system.update(dt=0.005)
            system.rescale_velocities(T0)

        # Сбор статистики
        stats = ExtendedStatsCollector()
        for _ in tqdm(range(20)):
            system.update(dt=0.005)
            stats.record(system)

        summary = stats.summarize()
        E_mean = summary["mean_E"] / N
        P_virial = summary["mean_P"]
        P_kinetic = np.mean([MASS * np.dot(p.vel, p.calc_force() / MASS) for p in system.particles]) / (L**2)

        E_list.append(E_mean)
        P_virial_list.append(P_virial)
        P_kinetic_list.append(P_kinetic)

        print(f"ρ = {rho:.2f} | E/N = {E_mean:.4f} | P_virial = {P_virial:.4f} | P_kinetic = {P_kinetic:.4f}")

    # --- Графики ---
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Энергия E/N(ρ)", "Давление P(ρ)"))

    fig.add_trace(go.Scatter(x=rhos, y=E_list, mode='lines+markers', name='E/N', line=dict(color="white")), row=1, col=1)

    fig.add_trace(go.Scatter(x=rhos, y=P_virial_list, mode='lines+markers', name='P (вириал)', line=dict(color="blue")), row=1, col=2)
    fig.add_trace(go.Scatter(x=rhos, y=P_kinetic_list, mode='lines+markers', name='P (импульс)', line=dict(color="red")), row=1, col=2)

    fig.update_layout(
        title="Зависимость энергии и давления от плотности при T = 0.45",
        template="plotly_dark",
        legend_title="Метод"
    )

    fig.update_xaxes(title_text="Плотность ρ", row=1, col=1)
    fig.update_yaxes(title_text="E/N", row=1, col=1)

    fig.update_xaxes(title_text="Плотность ρ", row=1, col=2)
    fig.update_yaxes(title_text="P", row=1, col=2)

    fig.show()


task3_pressure_vs_density()