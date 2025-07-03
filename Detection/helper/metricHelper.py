def update_metric(shared_metrics, key: str, value: float):
    arr = shared_metrics[key]
    arr.append(value)
    # If we've exceeded maxlen, remove from the front
    if len(arr) > 120:
        del arr[0:len(arr) - 120]
