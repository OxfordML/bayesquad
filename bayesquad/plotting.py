"""Functions to allow plotting code to be decoupled from the rest of the code."""

from functools import wraps
from types import MappingProxyType
from typing import Callable


class _Registry:
    def __init__(self):
        self._callback_registry = {}

    def trigger_callbacks(self, identifier: str, func: Callable):
        if identifier not in self._callback_registry:
            return

        for callback in self._callback_registry[identifier]:
            callback(func)

    def add_callback(self, identifier: str, callback: Callable):
        if identifier not in self._callback_registry:
            self._callback_registry[identifier] = []

        self._callback_registry[identifier].append(callback)


_function_registry = _Registry()

# Using a mutable object (e.g. an empty dict) as a default parameter can lead to undesirable behaviour, so we use this
# read-only proxy.
#
# See:
#   The problem: https://stackoverflow.com/q/1132941
#   A solution:  https://stackoverflow.com/a/30638022
_EMPTY = MappingProxyType({})


def plottable(identifier: str, *, default_plotting_parameters=_EMPTY):
    def decorator(func: Callable):
        @wraps(func)
        def func_for_plotting(*args, **kwargs):
            # Merge default_plotting_parameters into kwargs
            kwargs = {**default_plotting_parameters, **kwargs}
            return func(*args, **kwargs)

        _function_registry.trigger_callbacks(identifier, func_for_plotting)

        return func
    return decorator


def add_callback(identifier: str, callback: Callable):
    _function_registry.add_callback(identifier, callback)
