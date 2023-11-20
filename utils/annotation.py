

def httpx_timeout_retry(max_retries):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RuntimeError as e:
                    continue
            raise RuntimeError(f"{func.__name__} exceeded maximum {max_retries} retries.")

        return wrapper

    return decorator

