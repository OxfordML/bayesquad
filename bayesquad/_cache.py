"""Basic caching functionality."""
from functools import wraps

_cache = {}


def last_value_cache(func):
    """Cache the result of most recent invocation of this method.

    This decorator may be applied to a method which takes one argument (excluding `self`). If the method is called
    consecutively with the same argument, the method will immediately return the previous result rather than computing
    the result again. Note that by "the same argument" we mean the same object - two different but equal objects will
    not be regarded as the same by this decorator.

    The cache is not shared between different instances of the same class.

    Warnings
    --------
    Instances of a class with at least one method using this decorator **must** have :func:`~clear_last_value_caches`
    called on them when they are destroyed (e.g. in the class's `__del__` method). If this is not done, a new instance
    with the same id may incorrectly share the destroyed instance's cache.

    Examples
    --------
    >>> import numpy as np

    >>> class Foo:
    ...     def __init__(self):
    ...         self._count_invocations = 0
    ...
    ...     def __del__(self):
    ...         clear_last_value_caches(self)
    ...
    ...     @last_value_cache
    ...     def do_something_expensive(self, array):
    ...         # Do something expensive here.
    ...
    ...         self._count_invocations += 1
    ...
    ...     def count_expensive_operations(self):
    ...         return self._count_invocations

    >>> foo = Foo()
    >>> a = np.array(1)
    >>> b = np.array(1)

    `a` and `b` are distinct:

    >>> a is b
    False

    Passing `a` twice in succession will hit the cache:

    >>> foo.do_something_expensive(a)
    >>> foo.count_expensive_operations()
    1

    >>> foo.do_something_expensive(a)
    >>> foo.count_expensive_operations()
    1

    We get a cache miss when passing a different object:

    >>> foo.do_something_expensive(b)
    >>> foo.count_expensive_operations()
    2

    Since only a single function call is cached, we get a cache miss when passing the original object again:

    >>> foo.do_something_expensive(a)
    >>> foo.count_expensive_operations()
    3

    The cache is not shared between instances:

    >>> bar = Foo()
    >>> bar.do_something_expensive(a)
    >>> bar.count_expensive_operations()
    1

    The following is a hack to stop PyCharm wrongly warning about unresolved references in this doctest.
    See https://youtrack.jetbrains.com/issue/PY-31517

    >>> self = Foo()
    """
    @wraps(func)
    def transformed_function(self, x):
        cache_key = "{}_{}".format(id(self), id(func))

        if cache_key not in _cache or x is not _cache[cache_key][0]:
            ret = func(self, x)

            _cache[cache_key] = [x, ret]

            return ret
        else:
            return _cache[cache_key][1]

    return transformed_function


def clear_last_value_caches(obj):
    """Clear the :func:`~last_value_cache` of every method on the given object.

    See Also
    --------
    :func:`~last_value_cache`"""
    obj_id = str(id(obj))

    keys_to_delete = []

    for key in _cache:
        if key.startswith(obj_id):
            keys_to_delete.append(key)

    for key in keys_to_delete:
        del _cache[key]
