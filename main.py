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

        for p in self.pclsys.particles:
            # r - расстояние между p и self
            r = self.cord - p.cord
            r_abs = np.sqrt(np.sum(r**2))
            if r_abs == 0.0:
                continue

            F += - (24 * ep / sigma**2) * ((2 * (sigma / r_abs)**14) - (sigma / r_abs)**8) * r

        return F / self.m
    
    def update(self, dt):
        acc = self.calc_acceleration()
        # Используем метод Верле:
        self.cord = self.cord + self.vel * dt + 1/2 * acc * dt**2
        self.vel = self.vel + 1/2 * (acc + self.calc_acceleration()) * dt

class ParticleSystem():
    def __init__(self, N, ep, sigma, m):
        self.particles = [Particle(self, ep, sigma, m) for _ in range(N)]
        self.time = 0.0
    def update(self, dt):
        for particle in self.particles:
            particle.update(dt)
        self.time += dt
