try:
    raise TimeoutException("Timeout")
except TimeoutError as e:
    print(e)