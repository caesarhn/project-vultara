import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameter fisika
g = 9.81           # gravitasi
dt = 0.01          # waktu per langkah
t_max = 10
steps = int(t_max / dt)

# Properti persegi
L = 1.0            # panjang sisi
m = 1.0            # massa
I = (1/6) * m * L**2  # momen inersia untuk persegi di pusat

# Kondisi awal
x = 0.0
y = 5.0            # posisi awal di atas tanah
vx = 0.0
vy = 0.0
theta = np.pi / 6  # sudut awal (30 derajat)
omega = 0.0        # kecepatan sudut

# Pantulan saat menyentuh tanah
bounce_factor = -0.6  # kecepatan vertikal dikalikan ini setelah pantul

# Animasi setup
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-0.5, 6)
square, = ax.plot([], [], 'k-')

def get_corners(xc, yc, theta):
    # Mendapatkan 4 titik sudut dari persegi setelah rotasi
    half = L / 2
    corners = np.array([
        [-half, -half],
        [ half, -half],
        [ half,  half],
        [-half,  half],
        [-half, -half]  # kembali ke awal
    ])
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    rotated = corners @ rot.T
    rotated[:, 0] += xc
    rotated[:, 1] += yc
    return rotated

def update(frame):
    global x, y, vx, vy, theta, omega

    # Gaya dan percepatan
    ay = -g
    alpha = 0  # tanpa torsi eksternal

    # Update translasi
    vy += ay * dt
    y += vy * dt

    # Deteksi tanah
    min_y = y - (L / 2) * abs(np.cos(theta))
    if min_y < 0:
        y -= min_y  # koreksi agar tidak tembus
        vy *= bounce_factor
        omega *= bounce_factor  # energi rotasi juga dikurangi

    # Update rotasi
    omega += alpha * dt
    theta += omega * dt

    # Gambar ulang persegi
    coords = get_corners(x, y, theta)
    square.set_data(coords[:, 0], coords[:, 1])
    return square,

ani = animation.FuncAnimation(fig, update, frames=range(steps), interval=10)
plt.title("Simulasi Persegi Jatuh (Rigid Body)")
plt.gca().set_aspect('equal')
plt.show()
