import numpy as np
from math import cos, sin, pi
from scipy.integrate import odeint
from matplotlib import pyplot as plt

R = 0.0065
L = 0.6
M = 0.06385
IR = 0.022530346
J = M * IR ** 2
G = 9.8
B = 10 ** -5
K = 10 ** -2


THETA_0 = 0
PHI_0 = -L / R
THETA_DOT_0 = 0
PHI_DOT_0 = 0


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


def get_second_der_matrix(theta: float, phi: float) -> np.ndarray:
    return np.array(
        [
            [M * (L - q_val(theta, phi)) ** 2, M * R * L * sin(beta(theta, phi) - theta)],
            [M * R * L * sin(beta(theta, phi) - theta), M * R ** 2 + J]
        ]
    )


def get_first_der_matrix(theta: float, phi: float, theta_dot: float, phi_dot: float) -> np.ndarray:
    beta_ = beta(theta, phi)
    beta_dot_ = beta_dot(theta, phi, theta_dot, phi_dot)
    q_dot = dot_q_val(theta, phi, theta_dot, phi_dot)
    q_ = q_val(theta, phi)
    return np.array(
        [
            [2 * M * (L - q_) * q_dot, M * R * L * (beta_dot_ - theta_dot) * cos(beta_ - theta)],
            [M * R * L * (beta_dot_ - theta_dot) * cos(beta_ - theta), 0]
        ]
    )


def get_common_forces(theta: float, phi: float, theta_dot: float, phi_dot: float) -> np.ndarray:
    vbx, vby = dot_r_b(theta, phi, theta_dot, phi_dot)
    q_theta = -(L - q_val(theta, phi)) * (M * G * sin(theta) + K * (vbx * cos(theta) + vby * sin(theta)))
    beta_ = beta(theta, phi)
    q_phi = R * (M * G * cos(beta_) - K * (vbx * sin(beta_) - vby * cos(beta_))) - B * phi_dot
    return np.array([q_theta, q_phi])


def get_d_t_dc(theta: float, phi: float, theta_dot: float, phi_dot: float) -> np.ndarray:
    beta_ = beta(theta, phi)
    q_ = q_val(theta, phi)
    dt_d_theta = -d_q_d_theta(theta, phi) * M * (L - q_) * theta_dot ** 2 + \
        (d_beta_d_theta(theta, phi) - 1) * M * R * L * theta_dot * phi_dot * cos(beta_ - theta)
    dt_d_phi = -d_q_d_phi(theta, phi) * M * (L - q_) * theta_dot ** 2 + \
        d_beta_d_phi(theta, phi) * M * R * L * theta_dot * phi_dot * cos(beta_ - theta)
    return np.array([dt_d_theta, dt_d_phi])


def get_double_dots(theta: float, phi: float, theta_dot: float, phi_dot: float) -> np.ndarray:
    fdm = get_first_der_matrix(theta, phi, theta_dot, phi_dot)
    sdm = get_second_der_matrix(theta, phi)
    common_forces = get_common_forces(theta, phi, theta_dot, phi_dot)
    d_t_dc = get_d_t_dc(theta, phi, theta_dot, phi_dot)

    left_side = common_forces + d_t_dc - fdm.dot(np.array([theta_dot, phi_dot]))
    try:
        return np.linalg.inv(sdm).dot(left_side)
    except np.linalg.LinAlgError:
        return np.array([0, left_side[1] / (M * R ** 2 + J)])


def r_a(theta: float, phi: float) -> np.ndarray:
    return (L - q_val(theta, phi)) * np.array([sin(theta), -cos(theta)])


def r_b(theta: float, phi: float) -> np.ndarray:
    beta_ = beta(theta, phi)
    return r_a(theta, phi) - R * np.array([cos(beta_), sin(beta_)])


def r_a_0(theta: float, phi: float) -> np.ndarray:
    return r_b(theta, phi) + R * np.array([cos(phi), sin(phi)])


def dot_r_a(theta: float, phi: float, theta_dot: float, phi_dot: float) -> np.ndarray:
    return (L - q_val(theta, phi)) * theta_dot * np.array([cos(theta), sin(theta)]) + \
        dot_q_val(theta, phi, theta_dot, phi_dot) * np.array([-sin(theta), cos(theta)])


def dot_r_b(theta: float, phi: float, theta_dot: float, phi_dot: float) -> np.ndarray:
    beta_ = beta(theta, phi)
    beta_dot_ = beta_dot(theta, phi, theta_dot, phi_dot)
    return dot_r_a(theta, phi, theta_dot, phi_dot) + R * beta_dot_ * np.array([sin(beta_), -cos(beta_)])


def kin_energy(theta: float, phi: float, theta_dot: float, phi_dot: float) -> float:
    q_ = q_val(theta, phi)
    beta_ = beta(theta, phi)
    return 0.5 * M * ((L - q_) * theta_dot) ** 2 + M * R * L * theta_dot * phi_dot * sin(beta_ - theta) + \
        0.5 * M * (R * phi_dot) ** 2 + 0.5 * J * phi_dot ** 2


def equation(state: np.ndarray, t: float) -> np.ndarray:
    theta, phi, theta_dot, phi_dot = state
    theta_ddot, phi_ddot = get_double_dots(theta, phi, theta_dot, phi_dot)
    return np.array([theta_dot, phi_dot, theta_ddot, phi_ddot])


def solve_dif_equations(t_0: float, t_1: float, state_0: np.ndarray,
                        steps_per_second: int = 2000):
    n = int(max(1.0, t_1 - t_0) * steps_per_second)
    timestamps = np.linspace(t_0, t_1, n)
    return odeint(equation, state_0, timestamps), timestamps


def plot_dependency(x_states: np.ndarray, y_states: np.ndarray, title: str,
                    label: str, color_: str, x_label: str, y_label: str):
    plt.title(title)
    plt.plot(x_states, y_states, color_, label=label, linewidth=1.0)
    plt.grid(color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    plt.grid(True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='best')
    plt.show()


def plot_sample():
    t_0 = 0
    t_1 = 15
    state_0 = np.array([THETA_0, PHI_0, THETA_DOT_0, PHI_DOT_0])
    solution, timestamps = solve_dif_equations(t_0, t_1, state_0)

    angles = solution[:, :2]
    r_bs = np.array([r_b(theta, phi) for theta, phi in angles])
    plot_dependency(timestamps, -r_bs[:, 1], '$y(t)$', r'$y(t)$', 'm', r'$t$', r'$y,m$')


def main_1():
    plot_sample()


def main_2():
    pass


if __name__ == '__main__':
    main_1()
