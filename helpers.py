from manim import *
from typing import Callable, Iterable, Sized, Optional, Union as TUnion


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def to_norm_fun(law: Callable[[float], np.ndarray]) -> Callable[[float], float]:
    return lambda ct: np.linalg.norm(law(ct))


def sign(v: float) -> float:
    if v > 0:
        return 1
    return -1


def bin_search(list_: type[Iterable, Sized], value: float) -> int:
    low = 0
    high = len(list_)

    while low != high:
        middle = (low + high) // 2
        if list_[middle] >= value:
            high = middle
        else:
            low = middle + 1

    return low


def c2p(axes: Axes, point: np.ndarray) -> np.ndarray:
    return axes.c2p(point[0], point[1], point[2])


def update_matrix(m: Matrix, values: Iterable):
    for mob, value in zip(m.get_entries(), values):
        mob.set_value(value)


def create_mut_obj(getter: Callable[[ValueTracker], Any],
                   tracker: ValueTracker,
                   updater: Callable = lambda getter, tracker: (lambda z: z.become(getter(tracker)))):
    obj = getter(tracker)
    obj.add_updater(updater(getter, tracker))
    return obj


def _get_vector(start: np.ndarray, end: np.ndarray,
                c: str, scale: float = 1.0, **kwargs) -> Arrow:
    return Arrow(
        start,
        start + (end - start) * scale,
        color=c,
        **kwargs
    )


def get_vector(t: ValueTracker, axes: Axes, c: str,
               start_function: Callable[[float], np.ndarray],
               end_function: Callable[[float], np.ndarray],
               scale: float, a_kwargs: dict = None) -> Arrow:
    res: Arrow = create_mut_obj(
        lambda tracker: _get_vector(
            c2p(axes, start_function(tracker.get_value())),
            c2p(axes, start_function(tracker.get_value()) + end_function(tracker.get_value())),
            c,
            scale,
            **a_kwargs if a_kwargs else dict()
        ),
        t
    )
    return res


def get_line(t: ValueTracker, axes: Axes, c: str,
             s_law: Callable[[float], np.ndarray], e_law: Callable[[float], np.ndarray],
             **kwargs):
    res: Line = create_mut_obj(
        lambda tracker: Line(
            c2p(axes, s_law(tracker.get_value())),
            c2p(axes, e_law(tracker.get_value())),
            buff=0,
            color=c,
            **kwargs
        ),
        t,
    )
    return res


def get_vector_title(t: ValueTracker, vector: Arrow, title: str, font_size: int,
                     shift_law: Callable[[np.ndarray], np.ndarray]) -> MathTex:
    res: MathTex = create_mut_obj(
        lambda tracker: MathTex(title, font_size=font_size).move_to(
            vector.get_end() + shift_law(vector.get_end() - vector.get_start())
        ),
        t,
        lambda getter, tracker:
        lambda z: z.move_to(vector.get_end() + shift_law(vector.get_end() - vector.get_start()))
    )
    return res


def get_dot_title(t: ValueTracker, axes: Axes, title: str, font_size: int,
                  law: Callable[[float], np.ndarray], shift_law: Callable[[float], np.ndarray],
                  c: str = WHITE) -> MathTex:
    res: MathTex = create_mut_obj(
        lambda tracker: MathTex(title, font_size=font_size, color=c).move_to(
            c2p(axes, law(tracker.get_value())) + shift_law(tracker.get_value())
        ),
        t,
        lambda getter, tracker:
        lambda z: z.move_to(c2p(axes, law(tracker.get_value())) + shift_law(tracker.get_value()))
    )
    return res


def get_mut_dot(t: ValueTracker, axes: Axes, c: str, radius: float, law: Callable[[float], np.ndarray]) -> Dot:
    return create_mut_obj(
        lambda tracker: Dot(
            c2p(axes, law(tracker.get_value())),
            color=c,
            radius=radius
        ),
        t,
        lambda getter, tracker: lambda z: z.move_to(c2p(axes, law(tracker.get_value())))
    )


def get_mut_circle(t: ValueTracker, axes: Axes, r: float, c: str, stroke_width: float,
                   center_law: Callable[[float], np.ndarray],
                   fill_opacity: float = 0) -> Circle:
    circle_r = np.linalg.norm(c2p(axes, np.array((r, 0, 0))) - c2p(axes, np.array((0, 0, 0))))
    return create_mut_obj(
        lambda tracker: Circle(
            circle_r, color=c, stroke_width=stroke_width, fill_color=c, fill_opacity=fill_opacity
        ).move_arc_center_to(
            c2p(axes, center_law(tracker.get_value()))
        ),
        t,
        lambda getter, tracker: lambda z: z.move_arc_center_to(
            c2p(axes, center_law(tracker.get_value()))
        )
    )


def _get_vec_val(t: ValueTracker, data: Callable[[float], TUnion[np.ndarray, float]],
                 data_configs: Optional[dict] = None, **kwargs) -> TUnion[DecimalMatrix, DecimalNumber]:
    data_configs = data_configs if data_configs else dict()
    initial_data = data(t.get_value())
    scale = kwargs.get('scale', 1.0)

    if isinstance(initial_data, np.ndarray):
        return create_mut_obj(
            lambda tracker: DecimalMatrix(
                [[scalar] for scalar in initial_data],
                **data_configs
            ).scale(scale),
            t,
            lambda getter, tracker: lambda z: update_matrix(z, data(tracker.get_value()))
        )
    else:
        return create_mut_obj(
            lambda tracker: DecimalNumber(
                initial_data,
                **data_configs
            ),
            t,
            lambda getter, tracker: lambda z: z.set_value(data(tracker.get_value()))
        )


def get_vec_info(t: ValueTracker, label: str, font_size: int,
                 data: Callable[[float], TUnion[np.ndarray, float]],
                 data_configs: Optional[dict] = None, **kwargs) -> VGroup:
    return VGroup(
        MathTex(label, font_size=font_size),
        _get_vec_val(t, data, data_configs, **kwargs)
    )


def get_axes_config(x_range: list, y_range: list, x_step: Optional[float], y_step: Optional[float],
                    font_size: int, x_length: float,
                    y_length: Optional[float] = None, stroke_width: int = 1, tip_width: float = 0.15,
                    tip_height: float = 0.15, tick_size: float = 0.03,
                    line_to_number_buff: float = MED_SMALL_BUFF / 3, x_decimal_place: int = 0,
                    y_decimal_place: int = 0,) -> dict:
    yx_scale = (y_range[1] - y_range[0]) / (x_range[1] - x_range[0])
    y_length = (x_length * yx_scale) if y_length is None else y_length

    if x_step is None:
        x_step = x_range[2]

    if y_step is None:
        y_step = y_range[2]

    return {
        'x_range': x_range,
        'y_range': y_range,
        'x_length': x_length,
        'y_length': y_length,
        'axis_config': {
            'stroke_width': stroke_width,
            'tip_width': tip_width,
            'tip_height': tip_height,
            'tick_size': tick_size,
            'font_size': font_size,
            'line_to_number_buff': line_to_number_buff,
        },
        'x_axis_config': {
            'numbers_to_include': np.arange(*[-(x_range[0] // -x_step) * x_step, x_range[1], x_step]),
            'decimal_number_config': {
                "num_decimal_places": x_decimal_place,
            },
        },
        'y_axis_config': {
            'numbers_to_include': np.arange(*[-(y_range[0] // -y_step) * y_step, y_range[1], y_step]),
            'decimal_number_config': {
                "num_decimal_places": y_decimal_place,
            },
        },
    }


def cut(law: Callable[[float], np.ndarray], from_: int = 0, to_: int = -1) -> Callable[[float], np.ndarray]:
    return lambda c_t: law(c_t)[from_:to_]


def sub_slice(law: Callable[[float], np.ndarray], idx: int) -> Callable[[float], float]:
    return lambda c_t: law(c_t)[idx]


def part_apply(law: Callable[[float, ...], Any], *args) -> Callable[[float], Any]:
    return lambda c_t: law(c_t, *args)


def flatten(list_: list) -> list:
    return [item for sublist in list_ for item in sublist]


def create_graph_law(law: Callable[[float], np.ndarray], idx: int) -> Callable[[float], np.ndarray]:
    return lambda ct: np.array(
        (ct, law(ct)[idx], 0)
    )


def deep_arrange(group: VGroup, direction: np.ndarray, buff: float, center: bool, aligned_edge: np.ndarray):
    group.arrange(direction, buff, center, aligned_edge=aligned_edge)
    for sub1, sub2 in zip(group.submobjects, group.submobjects[1:]):
        for m1, m2 in zip(sub1.submobjects[1:], sub2.submobjects[1:]):
            m2.next_to(m1, direction, buff, aligned_edge=aligned_edge)
    return group


def get_timer(t: ValueTracker, timer_pos: np.ndarray, font_size: int,
              buff: float = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER / 5,
              num_decimal_places: int = 3) -> VGroup:
    return get_vec_info(
        t, 't=', font_size, lambda c_t: c_t, {'num_decimal_places': num_decimal_places, 'font_size': font_size}
    ).arrange(RIGHT, buff=buff).move_to(timer_pos)


def find_map(from_: np.ndarray, to_: np.ndarray,
             val: Optional[float] = None, idx: Optional[int] = None) -> tuple[int, float]:
    if idx is not None:
        return idx, to_[idx]
    if val is None:
        raise ValueError('At least val should not be None, when idx is None')

    idx = bin_search(from_, val)
    cur_v, prev_v = val, val
    cur_m, pred_m = 0, 0

    if idx < len(from_):
        cur_m, cur_v = to_[idx], from_[idx]
    if idx > 0:
        pred_m, prev_v = to_[idx - 1], from_[idx - 1]

    if idx == 0:
        return idx, cur_m
    if idx == len(from_):
        return idx, pred_m

    return idx, (cur_m - pred_m) * (val - prev_v) / (cur_v - prev_v) + pred_m

