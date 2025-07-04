def update_metric(shared_metrics, key: str, value: float):
    arr = shared_metrics[key]
    arr.append(value)
    if len(arr) > 120:
        arr[:] = arr[-120:]
