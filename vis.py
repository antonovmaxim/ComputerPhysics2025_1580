import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
np.random.seed(42)

# === Безразмерные параметры ===
EPSILON = 1.0
SIGMA = 1.0
MASS = 1.0
T0 = 1 #?
# Постоянная Больцмана
K_b = 1

class Particle:
    def __init__(self, pclsys, cord):
        self.pclsys = pclsys
        self.cord = cord.copy()
        self.vel = np.array([0.0, 0.0])  # будет задана позже

    def calc_acceleration(self):
        F = np.array([0.0, 0.0])
        L = self.pclsys.L
        sigma = SIGMA
        eps = EPSILON

        for p in self.pclsys.particles:
            if p is self:
                continue

            r = self.cord - p.cord
            r -= L * np.round(r / L)  # периодические граничные условия
            r_abs = np.linalg.norm(r)
            if r_abs == 0.0:
                continue

            f_scalar = (24 * eps / r_abs**2) * ((2 * (sigma / r_abs)**12) - (sigma / r_abs)**6)
            F += f_scalar * r

        return F / MASS

    def calc_Ek(self):
        return 0.5 * MASS * np.dot(self.vel, self.vel)

    def calc_Ep(self):
        def V(r):
            return 4 * EPSILON * ((SIGMA / r)**12 - (SIGMA / r)**6)

        ep = 0.0
        L = self.pclsys.L

        for p in self.pclsys.particles:
            if p is self:
                continue
            r = self.cord - p.cord
            r -= L * np.round(r / L)
            r_abs = np.linalg.norm(r)
            if r_abs == 0.0:
                continue
            ep += V(r_abs)

        return ep

class ParticleSystem:
    def __init__(self, N, L):
        self.N = N
        self.L = L
        self.particles = []

        # --- Располагаем частицы в узлах решетки ---
        n_side = int(np.sqrt(N))
        a = L / n_side

        for i in range(n_side):
            for j in range(n_side):
                if len(self.particles) >= N:
                    break
                x = (i + 0.5) * a
                y = (j + 0.5) * a
                self.particles.append(Particle(self, np.array([x, y])))

        # --- Случайные начальные скорости ---
        for p in self.particles:
            p.vel = np.random.randn(2)  # нормальное распределение

        # --- Удаляем скорость центра масс ---
        v_cm = sum(p.vel for p in self.particles) / N
        for p in self.particles:
            p.vel -= v_cm

        self.time = 0.0

    def update(self, dt):
        # 1. Старые ускорения
        acc_old = [p.calc_acceleration() for p in self.particles]

        # 2. Обновляем координаты
        for i, p in enumerate(self.particles):
            p.cord += p.vel * dt + 0.5 * acc_old[i] * dt**2
            p.cord = p.cord % self.L

        # 3. Новые ускорения
        acc_new = [p.calc_acceleration() for p in self.particles]

        # 4. Обновляем скорости
        for i, p in enumerate(self.particles):
            p.vel += 0.5 * (acc_old[i] + acc_new[i]) * dt

        # 5. Термализация
        # Простая термализация с масштабированием скоростей
        if self.time < 0.005*20:  # термализация только в начале
            for p in self.particles:
                p.vel *= np.sqrt(2*K_b*T0/(MASS*p.vel.dot(p.vel)))

        self.time += dt

    def calc_Ek(self):
        return sum(p.calc_Ek() for p in self.particles)

    def calc_Ep(self):
        return sum(p.calc_Ep() for p in self.particles) / 2

    def calc_E(self):
        return self.calc_Ek() + self.calc_Ep()

N = 100
L = 10.0         # Размер ящика (в единицах σ)
dt = 0.005       # Безразмерный шаг
n_steps = 200    # Кол-во шагов

system = ParticleSystem(N, L)

frames = []
for _ in tqdm(range(n_steps)):
    system.update(dt)
    x = [p.cord[0] for p in system.particles]
    y = [p.cord[1] for p in system.particles]
    frames.append(go.Frame(data=[go.Scatter(x=x, y=y, mode='markers')]))

fig = go.Figure(
    data=[go.Scatter(x=[], y=[], mode='markers')],
    layout=go.Layout(
        xaxis=dict(range=[0, system.L]),
        yaxis=dict(range=[0, system.L]),
        title="Молекулы в коробке",
        updatemenus=[dict(type="buttons", buttons=[dict(label="▶️", method="animate", args=[None])])],
        template='plotly_dark'
    ),
    frames=frames
)




fig.show()
