import numpy as np

def random_meal_scenario(
    rng=None,
    n_meals_range=(0, 7),       # inclusive; 0 allows fasting days
    carb_range=(15, 110),       # grams per meal
    time_window=(0.25, 23.0),   # hours after sim start a meal can occur
    min_gap=0.5,                # min hours between consecutive meals
    max_daily_carbs=None,       # optional cap on total CHO/day, e.g. 300
):
    """One random meal scenario as a sorted list of (time_h, carbs_g).
    Returns [] for a fasting day."""
    rng = np.random.default_rng() if rng is None else rng

    n = int(rng.integers(n_meals_range[0], n_meals_range[1] + 1))
    if n == 0:
        return []

    lo, hi = time_window
    # place meals with a minimum spacing (rejection sampling; trivial for n<=7)
    times = np.sort(rng.uniform(lo, hi, size=n))
    for _ in range(50):
        if n == 1 or np.all(np.diff(times) >= min_gap):
            break
        times = np.sort(rng.uniform(lo, hi, size=n))
    else:
        times = np.linspace(lo, hi, n)          # fallback if window too tight

    carbs = rng.uniform(carb_range[0], carb_range[1], size=n)
    if max_daily_carbs is not None and carbs.sum() > max_daily_carbs:
        carbs *= max_daily_carbs / carbs.sum()  # scale down proportionally

    times = np.round(times, 1)
    carbs = np.round(carbs).astype(int)
    return [(float(t), int(c)) for t, c in zip(times, carbs)]