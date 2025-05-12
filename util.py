from functools import wraps


def round_return(ndigits: int | None = None):
    """Decorator to round the return value of a function.

    Parameters
    ----------
    ndigits: int, optional
        Number of digits to round to. Default is None.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, (int, float)):
                return round(result, ndigits)
            elif isinstance(result, tuple):
                return tuple(
                    round(x, ndigits)
                    if isinstance(x, (int, float))
                    else x for x in result)
            elif isinstance(result, list):
                return [
                    round(x, ndigits)
                    if isinstance(x, (int, float))
                    else x for x in result]
            elif isinstance(result, dict):
                 return {
                    k: (round(v, ndigits)
                    if isinstance(v, (int, float))
                    else v) for k, v in result.items()}
            return result
        return wrapper
    return decorator
