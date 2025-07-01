import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def f1(gamma, lambd):
    def rhs(t, X):
        fi, y = X
        return [y, gamma - lambd * y - np.sin(fi)]
    return rhs


def eq_quiver(rhs, limits, N=16):
    fi_lims, y_lims = limits
    fis = np.linspace(fi_lims[0], fi_lims[1], N)
    ys = np.linspace(y_lims[0], y_lims[1], N)
    U = np.zeros((N, N))
    V = np.zeros((N, N))
    for i, y in enumerate(ys):
        for j, fi in enumerate(fis):
            vfield = rhs(0., [fi, y])
            u, v = vfield
            U[i][j] = u
            V[i][j] = v
    return fis, ys, U, V


def plot_plane(rhs, limits):
    fi_lims, y_lims = limits
    plt.xlim(fi_lims[0], fi_lims[1])
    plt.ylim(y_lims[0], y_lims[1])
    fi_vec, y_vec, U, V = eq_quiver(rhs, limits)
    plt.quiver(fi_vec, y_vec, U, V, alpha=0.8)


def point_okr(x, x0, err):
    return x0-err < x < x0+err


def turnover_made(t, y):
    return y[0] - np.pi
turnover_made.terminal = True
# turnover_made.direction = 1


# for y0 in np.arange(0., 2.5, 0.01):
#     break_flag = 0
#
#     rhs = f1(gamma, lambd)
#     # events = [turnover_completed]
#     sol1 = solve_ivp(rhs, [0, 5], (-np.pi, y0), method='RK45', rtol=1e-6)
#
#     fis, ys = sol1.y
#
#     for k in range(len(fis)):
#         if point_okr(ys[k], y0, y_err) and point_okr(fis[k], np.pi, fi_err):
#             print('y0 = ', y0, '; yk = ', ys[k], '; fik = ', fis[k], sep='')
#             break_flag = 1
#             break
#
#     if break_flag: break


def g(y):
    sol = solve_ivp(rhs, [0, 10], (-np.pi, y), method='RK45', rtol=1e-6, max_step=0.01, events=turnover_made)
    fis, ys = sol.y
    return ys[-1] - y


def g_1st_der(y):
    delta_y = 0.000001
    return (g(y+delta_y) - g(y)) / delta_y


def find_yn(y):
    return y - g(y) / g_1st_der(y)

def f(y):
    sol = solve_ivp(rhs, [0, 10], (-np.pi, y), method='RK45', rtol=1e-6, max_step=0.01, events=turnover_made)
    fis, ys = sol.y
    return ys[-1]


def f_1st_der(y):
    delta_y = 0.000001
    return (f(y+delta_y) - f(y)) / delta_y


y0 = 0.5
fi0 = -np.pi
gamma = 1.01
lambd = 0.5
rhs = f1(gamma, lambd)

y_err = 0.00001
# fi_err = 0.5

yi = y0
for i in range(100):
    yi = find_yn(yi)

    sol = solve_ivp(rhs, [0, 10], (fi0, yi), method='RK45', rtol=1e-6, max_step=0.01, events=turnover_made)
    fis, ys = sol.y

    # if i % 100 == 0:
    #     print(i, ': y0-yk = ', yi - ys[-1])

    if point_okr(ys[-1], yi, y_err):
        # print('\n', i, ':\ny0 = ', yi, '; yk = ', ys[-1], '; fik = ', fis[-1], '\ny0-yk = ', yi - ys[-1], sep='')
        break
    # fis, ys = sol1.y
    # for k in range(len(fis)):
    #     if point_okr(ys[k], yi, y_err) and point_okr(fis[k], np.pi, fi_err):
    #         # print('\n', i, ': y0 = ', y0, '; yk = ', ys[k], '; fik = ', fis[k], sep='')
    #         print('\n', i, ':\ny0 = ', yi, '; yk = ', ys[k], '; fik = ', fis[k], '\ny0-yk = ', yi - ys[k], sep='')
    #         break_flag = 1
    #         break
    #     elif fis[k] >= np.pi:
    #         # print(i, ': y0 = ', yi, '; yk = ', ys[k], '; fik = ', fis[k], sep='')
    #         # print(i, ': y0-yk = ', yi - ys[k], '; fik = ', fis[k], sep='')
    #         break

    # if break_flag:
    #     break

# y0 = 1.51
# rhs = f1(gamma, lambd)
# plt.axhline(y=y0, color='red', linestyle='-', linewidth=0.5)
# sol1 = solve_ivp(rhs, [0, 5], (fi0, y0), method='RK45', rtol=1e-6, max_step=0.01, events=turnover_made)

plot_plane(rhs, [(-np.pi, np.pi), (0, 3)])
fi, y = sol.y
plt.plot(fi, y, 'b-')
plt.axhline(y=yi, color='red', linestyle='-', linewidth=0.5)

# print('\n', sol1.t)
# print('\n', sol1.y[0][-1], sol1.y[1][-1])

print('\n', i, ':\ny0 = ', yi, '; y_last = ', ys[-1], '\nerr = ', yi - ys[-1], sep='')
print(sol.t_events)
print(f_1st_der(yi))

plt.show()
