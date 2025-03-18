def timed_function(f, *args, **kwargs):
        def new_func(*args, **kwargs):
            import time
            t = time.ticks_us()
            result = f(*args, **kwargs)
            delta = time.ticks_diff(time.ticks_us(), t) 
            print('Function {} Time = {:6.3f}ms'.format(f.__name__, delta/1000))
            return result
        return new_func
