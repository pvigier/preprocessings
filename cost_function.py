import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def cost_function(W, X, Y):
    predicted_Y = np.dot(X, W)
    error = (predicted_Y.T - Y).T
    return np.sum(np.square(error), axis=0) / len(X)

def gradient_cost_function(w, X, Y):
    return 2*(np.linalg.multi_dot([X.T, X, w]) - np.dot(X.T, Y)) / len(X)

def find_best_solution(X, Y):
    return np.dot(np.linalg.pinv(X), Y)

def get_extent(w_best):
    return w_best[0]-3, w_best[0]+3, w_best[1]-2, w_best[1]+2

def get_grid(extent):
    w1_min, w1_max, w2_min, w2_max = extent
    w1s = np.linspace(w1_min, w1_max, n)
    w2s = np.linspace(w2_min, w2_max, n)
    w1s, w2s = np.meshgrid(w1s, w2s)
    w1s, w2s = w1s.reshape(1, w1s.size), w2s.reshape(1, w2s.size)
    return np.concatenate((w1s, w2s))

def descend_gradient(w0, X, Y, nb_steps):
    ws = [w0]
    for _ in range(nb_steps):
        ws.append(ws[-1] - learning_rate*gradient_cost_function(ws[-1], X, Y))
    return ws

def get_init_point(w, distance):
    angle = np.random.rand() * 2 * np.pi
    dw = np.array([np.cos(angle), np.sin(angle)]) * distance
    return w + dw

def plot_gradient_descent(ws1, ws2):
    xs1, ys1 = [], []
    xs2, ys2 = [], []
    line1 = ax1.plot([], [], 'r', animated=True)[0]
    line2 = ax2.plot([], [], 'r', animated=True)[0]

    def init():
        return line1, line2

    def update(frame):
        w1, w2, t = frame
        # line1
        xs1.append(w1[0])
        ys1.append(w1[1])
        line1.set_data(xs1, ys1)
        # line2
        xs2.append(w2[0])
        ys2.append(w2[1])
        line2.set_data(xs2, ys2)
        # Reset
        if t == nb_steps - 1:
            xs1.clear()
            ys1.clear()
            xs2.clear()
            ys2.clear()
        return line1, line2

    frames = list(zip(ws1, ws2, range(nb_steps)))
    return FuncAnimation(fig, update, init_func=init, frames=frames, blit=True, repeat=True)

def plot_contour(X, Y, ax, title):
    w_best = find_best_solution(X, Y)
    extent = get_extent(w_best)
    W = get_grid(extent)

    cost = cost_function(W, X, Y).reshape(n, n)
    #ax.imshow(cost, origin='lower', extent=extent)
    cp = ax.contour(cost, levels, extent=extent)
    ax.clabel(cp)
    ax.axis('scaled')
    ax.set_title(title)
    return w_best

# Create dataset
N = 100
X = np.concatenate((np.ones((N, 1)), np.random.randn(N, 1)*3+2), axis=1)
Y = 3*X[:,1] + 1

# Parameters for the contour plot
n = 100
levels = [0.1, 1, 3, 10, 30]

# Parameters for gradient descent
distance = 2
learning_rate = 0.07
nb_steps = 30

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2)

# Without preprocessings
w_best = plot_contour(X, Y, ax1, 'Without preprocessings')
w0 = get_init_point(w_best, distance)
ws1 = descend_gradient(w0, X, Y, nb_steps)

# With preprocessings
X = np.copy(X)
X[:,1] = (X[:,1] - np.mean(X[:,1])) / np.std(X[:,1])
w_best = plot_contour(X, Y, ax2, 'With preprocessings')
w0 = get_init_point(w_best, distance)
ws2 = descend_gradient(w0, X, Y, nb_steps)

# Create animation
ani = plot_gradient_descent(ws1, ws2)
ani.save('cost_function.gif', dpi=160, writer='imagemagick')

plt.show()