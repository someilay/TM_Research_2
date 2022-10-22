from manim import *
from typing import Optional
from helpers import get_axes_config, bin_search, get_mut_circle, get_mut_dot, get_dot_title, get_timer, \
    get_line, find_map
from main import R, r_a, r_a_0, r_b, solve_dif_equations


class Main(Scene):
    ANGLES: Optional[np.ndarray] = None
    THETAS: Optional[np.ndarray] = None
    PHIS: Optional[np.ndarray] = None

    ANGULAR_VS: Optional[np.ndarray] = None
    THETA_DOTS: Optional[np.ndarray] = None
    PHI_DOTS: Optional[np.ndarray] = None

    TIMESTAMPS: Optional[np.ndarray] = None

    INIT_T = 0
    FINAL_T = 10
    PLAYBACK_SPEED = 0.2
    STEPS = int(max(1.0, FINAL_T - INIT_T) * 2000)

    L = 0.4
    THETA_0 = 0
    PHI_0 = -L / R
    THETA_DOT_0 = 0
    PHI_DOT_0 = 0

    VECTORS_KWARGS = {
        'stroke_width': 2,
        'tip_length': 0.25,
        'buff': 0,
        'max_tip_length_to_length_ratio': 0.05,
        'max_stroke_width_to_length_ratio': 10,
    }

    @staticmethod
    def init_vals():
        source = Main
        if source.ANGLES is None or source.ANGULAR_VS is None:
            state_0 = np.array([source.THETA_0, source.PHI_0, source.THETA_DOT_0, source.PHI_DOT_0])
            solution, source.TIMESTAMPS = solve_dif_equations(source.INIT_T, source.FINAL_T, source.L, state_0)
            source.ANGLES = solution[:, :2]
            source.THETAS, source.PHIS = source.ANGLES[:, 0], source.ANGLES[:, 1]
            source.ANGULAR_VS = solution[:, 2:]
            source.THETA_DOTS, source.PHI_DOTS = source.ANGULAR_VS[:, 0], source.ANGULAR_VS[:, 1]

    @staticmethod
    def r_a(t: float) -> np.ndarray:
        source = Main
        source.init_vals()

        idx, theta = find_map(source.TIMESTAMPS, source.THETAS, val=t)
        _, phi = find_map(source.TIMESTAMPS, source.PHIS, idx=idx)

        r_ax, r_ay = r_a(theta, phi, source.L)
        return np.array([r_ax, r_ay, 0])

    @staticmethod
    def r_a_0(t: float) -> np.ndarray:
        source = Main
        source.init_vals()

        idx, theta = find_map(source.TIMESTAMPS, source.THETAS, val=t)
        _, phi = find_map(source.TIMESTAMPS, source.PHIS, idx=idx)

        r_ax_0, r_ay_0 = r_a_0(theta, phi, source.L)
        return np.array([r_ax_0, r_ay_0, 0])

    @staticmethod
    def r_b(t: float) -> np.ndarray:
        source = Main
        source.init_vals()

        idx, theta = find_map(source.TIMESTAMPS, source.THETAS, val=t)
        _, phi = find_map(source.TIMESTAMPS, source.PHIS, idx=idx)

        r_bx, r_by = r_b(theta, phi, source.L)
        return np.array([r_bx, r_by, 0])

    def construct(self):
        c_time = ValueTracker(self.INIT_T)
        axes = Axes(
            **get_axes_config(
                [-0.1, 0.1, 0.05],
                [-0.45, 0.05, 0.05],
                None,
                None,
                12,
                3,
                x_decimal_place=3,
                y_decimal_place=2
            )
        )

        a_dot = get_mut_dot(c_time, axes, WHITE, DEFAULT_DOT_RADIUS / 4, self.r_a)
        a_0_dot = get_mut_dot(c_time, axes, WHITE, DEFAULT_DOT_RADIUS / 4, self.r_a_0)
        b_dot = get_mut_dot(c_time, axes, WHITE, DEFAULT_DOT_RADIUS / 4, self.r_b)

        a_title = get_dot_title(c_time, axes, 'A', 12, self.r_a, lambda _: DOWN / 6)
        a_0_title = get_dot_title(c_time, axes, 'A_0', 12, self.r_a_0, lambda _: LEFT / 6)
        b_title = get_dot_title(c_time, axes, 'B', 12, self.r_b, lambda _: UP / 6)

        circle = get_mut_circle(c_time, axes, R, GREEN, DEFAULT_STROKE_WIDTH / 4, self.r_b, 0.5)
        line = get_line(c_time, axes, GREEN, lambda _: np.zeros(3), self.r_a, stroke_width=DEFAULT_STROKE_WIDTH / 3)

        timer = get_timer(c_time, LEFT * 3, 16, num_decimal_places=3)

        self.add(axes)
        self.add(circle, line)
        self.add(a_dot, a_0_dot, b_dot, a_title, a_0_title, b_title)
        self.add(timer)
        self.play(
            c_time.animate.set_value(self.FINAL_T),
            run_time=(self.FINAL_T - self.INIT_T) / self.PLAYBACK_SPEED,
            rate_func=linear
        )
