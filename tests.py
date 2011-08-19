from nose import with_setup

def setup_func():
    pass

def teardown_func():
    pass


@with_setup(setup_func, teardown_func)
def test():
    pass



