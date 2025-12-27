import numpy as np
import scipy.optimize

from .sysidalg import sysid


def cv(
    training_data,
    validation_data,
    y_name,
    u_name,
    order,
    method=None,
    settings=None,
    bounds=(0, 100),
):
    if settings is None:
        settings = {}

    def fun(lmb_exp):
        s = settings.copy()
        s["lambda"] = 10**lmb_exp
        mod = sysid(training_data, y_name, u_name, order, method=method, settings=s)
        y = validation_data[y_name]
        u = validation_data[u_name]
        fit = mod.compare(y, u)
        print(10**lmb_exp, -fit)
        return -fit

    bounds_log = (np.log10(max(bounds[0], 1e-12)), np.log10(max(bounds[1], 1e-12)))
    res = scipy.optimize.minimize_scalar(fun, bounds=bounds_log)

    return 10**res.x, -res.fun


def rise_time(mod):
    t, s = mod.step_response(N=200)
    ind_start = np.argwhere(s > 0.1)[0]
    ind_end = np.argwhere(s > 0.9)[0]
    t_rise = t[ind_end] - t[ind_start]
    return float(t_rise.item())


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
    return float(t_rise.item())
