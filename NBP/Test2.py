import numpy as np

n = 19
k = 7
n_particles = 29
kn_particles = k * n_particles
D = np.random.uniform(0, 500, n)
D[D > 30] = 0
neighbours = np.count_nonzero(D)
print(neighbours)
c = 0
for i in range(neighbours):
    m = kn_particles // neighbours
    c += m
    kn_particles -= m
    neighbours -= 1
    print(m)

print(kn_particles)