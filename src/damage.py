from rfcnt import rfcnt

def rfc_wrapper(signal_, upper, lower, class_count, residual_method, spread_damage):
    if upper is None:
        upper = max(signal_)
    if lower is None:
        lower = min(signal_)

    class_range = upper - lower
    class_width = class_range / (class_count - 1)
    class_offset = lower - class_width / 2
    
    RFC_KWARGS = {"class_count": class_count,
        "class_offset": class_offset,
        "class_width": 0.2,
        "hysteresis": class_width,
        "auto_resize": True,
        "use_HCM": 0,
        "use_ASTM": 1,
        "spread_damage": 8,
        "residual_method": residual_method, # For ASTM this should be 4, but with 4 it is not montonously increasing.
        "wl": {"sd": 865/2, "nd": 2e6, "k": 8}
    }
    class_range = upper - lower
    class_width = class_range / (class_count - 1)
    class_offset = lower - class_width / 2
    res = rfcnt.rfc(signal_, **RFC_KWARGS)
    return res
    
def calculate_damage(signal_, upper=None, lower=None, class_count=200, residual_method=0):
    damage = rfc_wrapper(signal_, upper, lower, class_count, residual_method, spread_damage=8)['damage']
    return damage

def calculate_damage_history(signal_, upper=None, lower=None, class_count=200, residual_method=0, spread_damage=8):
    dh = rfc_wrapper(signal_, upper, lower, class_count, residual_method, spread_damage=8)['dh']
    return dh