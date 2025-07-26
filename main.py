import numpy as np

class Particle():
    def __init__(self, pclsys, ep, sigma, m):
        self.cord = np.array([0.0, 0.0]) # в СИ, м
        self.vel = np.array([0.0, 0.0]) # в СИ, м/с
        self.pclsys = pclsys
        self.sigma = sigma
        self.ep = ep
        self.m = m

    def calc_acceleration(self):
        # params: pclsys, vel, cord
        ep = self.ep
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

class ParticleSystem():
    def __init__(self, N, ep, sigma, m, L):
        # L - длина стороны квадрата, в котором расположены частицы
        self.L = L
        self.particles = [Particle(self, ep, sigma, m) for _ in range(N)]
        self.time = 0.0

    def update(self, dt):
        for particle in self.particles:
            particle.update(dt)
        self.time += dt
