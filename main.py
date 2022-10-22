import json
import os
import numpy as np
from math import cos, sin, pi, inf
from scipy.integrate import odeint
from matplotlib import pyplot as plt

from helpers import find_map

R = 0.0065
M = 0.06385
IR = 0.022530346
J = M * IR ** 2
G = 9.8
B = 10 ** -5
K = 10 ** -2


def beta(theta: float, phi: float) -> float:
    return max(min(phi, pi + theta), theta)


def beta_dot(theta: float, phi: float, theta_dot: float, phi_dot: float) -> float:
    if theta <= phi < pi + theta:
        return phi_dot
    else:
        return theta_dot


def d_beta_d_theta(theta: float, phi: float) -> float:
    if theta <= phi < pi + theta:
        return 0.0
    else:
        return 1.0


def d_beta_d_phi(theta: float, phi: float) -> float:
    if theta <= phi < pi + theta:
        return 1.0
    else:
        return 0.0


def q_val(theta: float, phi: float) -> float:
    return R * abs(phi - beta(theta, phi))


def dot_q_val(theta: float, phi: float, theta_dot: float, phi_dot: float) -> float:
    if theta <= phi < pi + theta:
        return 0.0
    elif phi >= pi + theta:
        return R * (phi_dot - theta_dot)
    else:
        return R * (theta_dot - phi_dot)


def d_q_d_theta(theta: float, phi: float) -> float:
    if theta <= phi < pi + theta:
        return 0.0
    elif phi >= pi + theta:
        return -R
    else:
        return R


def d_q_d_phi(theta: float, phi: float) -> float:
    if theta <= phi < pi + theta:
        return 0.0
    elif phi >= pi + theta:
        return R
    else:
        return -R


def get_second_der_matrix(theta: float, phi: float, length: float) -> np.ndarray:
    return np.array(
        [
            [M * (length - q_val(theta, phi)) ** 2, M * R * length * sin(beta(theta, phi) - theta)],
            [M * R * length * sin(beta(theta, phi) - theta), M * R ** 2 + J]
        ]
    )


def get_first_der_matrix(theta: float, phi: float, theta_dot: float, phi_dot: float, length: float) -> np.ndarray:
    beta_ = beta(theta, phi)
    beta_dot_ = beta_dot(theta, phi, theta_dot, phi_dot)
    q_dot = dot_q_val(theta, phi, theta_dot, phi_dot)
    q_ = q_val(theta, phi)
    return np.array(
        [
            [2 * M * (length - q_) * q_dot, M * R * length * (beta_dot_ - theta_dot) * cos(beta_ - theta)],
            [M * R * length * (beta_dot_ - theta_dot) * cos(beta_ - theta), 0]
        ]
    )


def get_common_forces(theta: float, phi: float, theta_dot: float, phi_dot: float, length: float) -> np.ndarray:
    vbx, vby = dot_r_b(theta, phi, theta_dot, phi_dot, length)
    q_theta = -(length - q_val(theta, phi)) * (M * G * sin(theta) + K * (vbx * cos(theta) + vby * sin(theta)))
    beta_ = beta(theta, phi)
    q_phi = R * (M * G * cos(beta_) - K * (vbx * sin(beta_) - vby * cos(beta_))) - B * phi_dot
    return np.array([q_theta, q_phi])


def get_d_t_dc(theta: float, phi: float, theta_dot: float, phi_dot: float, length: float) -> np.ndarray:
    beta_ = beta(theta, phi)
    q_ = q_val(theta, phi)
    dt_d_theta = -d_q_d_theta(theta, phi) * M * (length - q_) * theta_dot ** 2 + \
        (d_beta_d_theta(theta, phi) - 1) * M * R * length * theta_dot * phi_dot * cos(beta_ - theta)
    dt_d_phi = -d_q_d_phi(theta, phi) * M * (length - q_) * theta_dot ** 2 + \
        d_beta_d_phi(theta, phi) * M * R * length * theta_dot * phi_dot * cos(beta_ - theta)
    return np.array([dt_d_theta, dt_d_phi])


def get_double_dots(theta: float, phi: float, theta_dot: float, phi_dot: float, length: float) -> np.ndarray:
    fdm = get_first_der_matrix(theta, phi, theta_dot, phi_dot, length)
    sdm = get_second_der_matrix(theta, phi, length)
    common_forces = get_common_forces(theta, phi, theta_dot, phi_dot, length)
    d_t_dc = get_d_t_dc(theta, phi, theta_dot, phi_dot, length)

    left_side = common_forces + d_t_dc - fdm.dot(np.array([theta_dot, phi_dot]))
    try:
        return np.linalg.inv(sdm).dot(left_side)
    except np.linalg.LinAlgError:
        return np.array([0, left_side[1] / (M * R ** 2 + J)])


def r_a(theta: float, phi: float, length: float) -> np.ndarray:
    return (length - q_val(theta, phi)) * np.array([sin(theta), -cos(theta)])


def r_b(theta: float, phi: float, length: float) -> np.ndarray:
    beta_ = beta(theta, phi)
    return r_a(theta, phi, length) - R * np.array([cos(beta_), sin(beta_)])


def r_a_0(theta: float, phi: float, length: float) -> np.ndarray:
    return r_b(theta, phi, length) + R * np.array([cos(phi), sin(phi)])


def dot_r_a(theta: float, phi: float, theta_dot: float, phi_dot: float, length: float) -> np.ndarray:
    return (length - q_val(theta, phi)) * theta_dot * np.array([cos(theta), sin(theta)]) + \
           dot_q_val(theta, phi, theta_dot, phi_dot) * np.array([-sin(theta), cos(theta)])


def dot_r_b(theta: float, phi: float, theta_dot: float, phi_dot: float, length: float) -> np.ndarray:
    beta_ = beta(theta, phi)
    beta_dot_ = beta_dot(theta, phi, theta_dot, phi_dot)
    return dot_r_a(theta, phi, theta_dot, phi_dot, length) + R * beta_dot_ * np.array([sin(beta_), -cos(beta_)])


def kin_energy(theta: float, phi: float, theta_dot: float, phi_dot: float, length: float) -> float:
    q_ = q_val(theta, phi)
    beta_ = beta(theta, phi)
    return 0.5 * M * ((length - q_) * theta_dot) ** 2 + M * R * length * theta_dot * phi_dot * sin(beta_ - theta) + \
        0.5 * M * (R * phi_dot) ** 2 + 0.5 * J * phi_dot ** 2


def equation(state: np.ndarray, t: float, length: float) -> np.ndarray:
    theta, phi, theta_dot, phi_dot = state
    theta_ddot, phi_ddot = get_double_dots(theta, phi, theta_dot, phi_dot, length)
    return np.array([theta_dot, phi_dot, theta_ddot, phi_ddot])


def solve_dif_equations(t_0: float, t_1: float, length: float, state_0: np.ndarray,
                        steps_per_second: int = 2000):
    n = int(max(1.0, t_1 - t_0) * steps_per_second)
    timestamps = np.linspace(t_0, t_1, n)
    return odeint(equation, state_0, timestamps, args=(length,)), timestamps


def extract_y(solution: np.ndarray, length: float) -> np.ndarray:
    r_bs = np.array([r_b(theta, phi, length) for theta, phi in solution[:, :2]])
    return -r_bs[:, 1]


def plot_dependency(x_states: list[np.ndarray], y_states: list[np.ndarray], title: str,
                    labels: list[str], colors: list[str], x_label: str, y_label: str, save_to: str):
    plt.title(title)
    for x_states_, y_states_, color_, label_ in zip(x_states, y_states, colors, labels):
        plt.plot(x_states_, y_states_, color_, label=label_, linewidth=1.0)
    plt.grid(color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    plt.grid(True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='best')
    plt.savefig('plots/' + save_to)
    plt.show()


def open_data(filepath: str) -> list:
    with open(filepath, 'rb') as data_file:
        return json.load(data_file)


def main():
    positions_file, timestamps_file = 'poscm.json', 'time.json'
    data = {}

    # Load data
    for path, dirs, files in os.walk('data'):
        if files:
            data[path] = [[], []]
        for file in files:
            if file == positions_file:
                data[path][0] = open_data(os.path.join(path, file))
            if file == timestamps_file:
                data[path][1] = open_data(os.path.join(path, file))

    # Sort data and put to buckets of 5 samples
    stack = []
    res = []
    for key in sorted(data.keys(), key=lambda x: max(data[x][0])):
        item = data[key]
        stack.append((key, item))

        if len(stack) == 5:
            mean = 0
            names = []
            times = []
            for name, elem in stack:
                names.append(name)
                times.append(elem[1][-1])
                mean += max(elem[0])
            # Calculate mean max y
            res.append((names, mean / 5, times))
            stack = [(key, item)]

    # For each bucket draw plots
    for samples, mean_length, times in res:
        y_states = []
        timestamps = []
        min_len = inf

        # Load data
        for sample in samples:
            y_states.append(data[sample][0])
            timestamps.append(data[sample][1])
            min_len = min(min_len, len(y_states[-1]), len(timestamps[-1]))
        min_len = int(min_len)

        # Align data
        for idx, (y_states_, timestamps_) in enumerate(zip(y_states, timestamps)):
            y_states[idx] = np.array(y_states_[:min_len]) / 100
            timestamps[idx] = np.array(timestamps_[:min_len])

        # Plot all samples of bucket
        plot_dependency(
            timestamps, y_states,
            f'Mean length: {mean_length:.0f}, All samples',
            samples,
            ['b', 'g', 'r', 'c', 'm'],
            '$t, s$', '$y, m$',
            f'all_samples_{mean_length:.0f}_cm.png'
        )

        # Calculate mean & std
        timestamps = np.mean(np.array(timestamps), axis=0)
        y_states_mean = np.mean(np.array(y_states), axis=0)
        y_states_std = np.std(np.array(y_states), axis=0)

        # Simulate
        length = mean_length / 100
        theta_0 = 0
        phi_0 = -(length - y_states_mean[0]) / R
        theta_dot_0 = 0
        phi_dot_0 = 0
        state_0 = np.array([theta_0, phi_0, theta_dot_0, phi_dot_0])
        solution, timestamps_2 = solve_dif_equations(timestamps[0], timestamps[-1], length, state_0)
        y_states_sim = extract_y(solution, length)

        # Plot STD
        plot_dependency(
            [timestamps], [y_states_std],
            f'Mean length: {mean_length:.0f} cm, STD',
            ['STD $y(t)$'],
            ['b', 'g', 'r', 'c', 'm'],
            '$t, s$', '$y, m$',
            f'std_{mean_length:.0f}_cm.png'
        )
        # Plot simulation and mean
        plot_dependency(
            [timestamps, timestamps_2], [y_states_mean, y_states_sim],
            f'Mean length: {mean_length:.0f} cm, Mean & Simulation',
            ['Mean $y(t)$', '$y_s(t)$'],
            ['b', 'g', 'r', 'c', 'm'],
            '$t, s$', '$y, m$',
            f'mean_and_sim_{mean_length:.0f}_cm.png'
        )

        # Calculate square deviation
        y_states_sim = np.array([find_map(timestamps_2, y_states_sim, val=ct)[1] for ct in timestamps])
        y_states_square = np.square(y_states_mean - y_states_sim) / mean_length ** 2

        # Plot norm square deviation
        plot_dependency(
            [timestamps], [y_states_square],
            f'Mean length: {mean_length:.0f} cm, $(mean - simulation)^2 / length^2$',
            ['$(y_{mean}(t) - y_s(t))^2 / length^2$'],
            ['b', 'g', 'r', 'c', 'm'],
            '$t, s$', 'error',
            f'norm_square_dev_{mean_length:.0f}_cm.png'
        )


if __name__ == '__main__':
    main()
