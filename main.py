import numpy as np

class Particle():
    def __init__(self, pclsys, eps, sigma, m, cord):
        self.cord = cord # в СИ, м
        self.vel = np.array([0.0, 0.0]) # в СИ, м/с
        self.pclsys = pclsys
        self.sigma = sigma
        self.eps = eps
        self.m = m

    def calc_acceleration(self):
        # params: pclsys, vel, cord
        ep = self.eps
        sigma = self.sigma
        F = np.array([0.0, 0.0])
        L = self.pclsys.L  # Размер квадрата

        for p in self.pclsys.particles:
            if p is self:
                continue

            # Рассчитываем расстояние с учетом периодических граничных условий
            r = self.cord - p.cord
            
            # Применяем периодические граничные условия для расчета кратчайшего расстояния
            r[0] = r[0] - L * round(r[0] / L)
            r[1] = r[1] - L * round(r[1] / L)
            
            r_abs = np.sqrt(np.sum(r**2))
            if r_abs == 0.0:
                continue

            F += - (24 * ep / sigma**2) * ((2 * (sigma / r_abs)**14) - (sigma / r_abs)**8) * r

        return F / self.m
    
    def update(self, dt):
        acc = self.calc_acceleration()
        # Используем метод Верле:
        self.cord = self.cord + self.vel * dt + 1/2 * acc * dt**2
        
        # Применяем периодические граничные условия
        L = self.pclsys.L
        self.cord[0] = self.cord[0] % L
        self.cord[1] = self.cord[1] % L
        
        self.vel = self.vel + 1/2 * (acc + self.calc_acceleration()) * dt
    
    def calc_Ek(self):
        return 0.5 * self.m * np.sum(self.vel**2)
    
    def calc_Ep(self):
        def V(r):
            return 4*self.eps*((self.sigma/r)**12 - (self.sigma/r)**6)
        
        ep = 0.0
        L = self.pclsys.L
        
        for p in self.pclsys.particles:
            if p is self:
                continue
                
            # Рассчитываем расстояние с учетом периодических граничных условий
            r = self.cord - p.cord
            
            # Применяем периодические граничные условия для расчета кратчайшего расстояния
            r[0] = r[0] - L * round(r[0] / L)
            r[1] = r[1] - L * round(r[1] / L)
            
            r_abs = np.sqrt(np.sum(r**2))
            if r_abs == 0.0:
                continue
            ep += V(r_abs)
        return ep
    

class ParticleSystem():
    def __init__(self, N, ep, sigma, m, L):
        # L - длина стороны квадрата, в котором расположены частицы
        self.L = L
        n_side = int(np.sqrt(N))  # Число частиц вдоль стороны квадрата
        a = L / n_side  # Расстояние между частицами

        self.particles = []
        # Размещаем частицы в регулярной решетке
        for i in range(n_side):
            for j in range(n_side):
                x = (i + 0.5) * a
                y = (j + 0.5) * a
                self.particles.append(Particle(self, ep, sigma, m, np.array([x, y])))
        
        # Если число частиц не является ровным квадратом - оставшиеся докинем случайно
        remaining = N - n_side * n_side
        for _ in range(remaining):
            x = np.random.uniform(0, L)
            y = np.random.uniform(0, L)
            self.particles.append(Particle(self, ep, sigma, m, np.array([x, y])))
            
        self.time = 0.0
        

    def update(self, dt):
        for particle in self.particles:
            particle.update(dt)
        self.time += dt
    
    def calc_Ek(self):
        return sum(p.calc_Ek() for p in self.particles)
    
    def calc_Ep(self):
        return sum(p.calc_Ep() for p in self.particles) / 2
    
    def calc_E(self):
        return self.calc_Ek() + self.calc_Ep()

# Пусть наш газ - Аргон
EPSILON = 1.66e-21 # Дж
SIGMA = 3.41e-10 # м
MASS = 6.63e-26 # кг

import plotly.graph_objects as go

def main():
    sys = ParticleSystem(100, EPSILON, SIGMA, MASS, 1e-8)
    dt = 1e-14
    ek_values = []
    ep_values = []
    e_values = []

    for i in range(40):
        sys.update(dt)
        k = sys.calc_Ek()
        p = sys.calc_Ep()
        ek_values.append(k)
        ep_values.append(p)
        e_values.append(k+p)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=ek_values, mode='lines', name='Кинетическая'))
    fig.add_trace(go.Scatter(y=ep_values, mode='lines', name='Потенциальная'))
    fig.add_trace(go.Scatter(y=e_values, mode='lines', name='Полная'))
    
    fig.update_layout(title='Энергия системы',
                      xaxis_title='Шаг по времени',
                      yaxis_title='Энергия (E)',
                      legend=dict(x=0, y=1))
    
    fig.show()
    
if __name__ == '__main__':
    main()