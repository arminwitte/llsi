from .sysidalg import sysid
import numpy as np
import scipy.optimize


def cv(
    training_data,
    validation_data,
    y_name,
    u_name,
    order,
    method=None,
    settings={},
    bounds=(0, 100),
):

    def fun(l):
        settings["lambda"] = 10**l
        mod = sysid(
            training_data, y_name, u_name, order, method=method, settings=settings
        )
        y = validation_data[y_name]
        u = validation_data[u_name]
        fit = mod.compare(y, u)
        print(10**l, -fit)
        return -fit

    bounds_log = (np.log10(max(bounds[0], 1e-12)), np.log10(max(bounds[1], 1e-12)))
    res = scipy.optimize.minimize_scalar(fun, bounds=bounds_log)

    return 10**res.x, -res.fun


def rise_time(mod):
    t, s = mod.step_response(N=200)
    ind_start = np.argwhere(s > 0.1)[0]
    ind_end = np.argwhere(s > 0.9)[0]
    t_rise = t[ind_end] - t[ind_start]
    return float(t_rise)


def settling_time(mod, margin=0.01):
    m_plus = 1.0 + margin
    m_minus = 1.0 - margin
    t, s = mod.step_response(N=200)
    index_over = np.argwhere(s > m_plus)
    index_under = np.argwhere(s < m_minus)
    if len(index_over) > 0:
        last_index_over = index_over[-1]
    else:
        last_index_over = 0
    if len(index_under) > 0:
        last_index_under = index_under[-1]
    else:
        last_index_under = 0
    last_index_outside_margin = max(last_index_over, last_index_under)
    t_rise = t[last_index_outside_margin]
    return float(t_rise)
