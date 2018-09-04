"""Functions to allow plotting code to be decoupled from the rest of the code."""

from functools import wraps
from typing import Callable


class Registry:
    def __init__(self):
        self._callback_registry = {}

    def update(self, identifier: str, func: Callable):
        if identifier not in self._callback_registry:
            return

        for callback in self._callback_registry[identifier]:
            callback(func)

    def add_callback(self, identifier: str, callback: Callable):
        if identifier not in self._callback_registry:
            self._callback_registry[identifier] = []

        self._callback_registry[identifier].append(callback)


_function_registry = Registry()


def plottable(identifier: str):
    def actual_decorator(func: Callable):
        _function_registry.update(identifier, func)
        return func
    return actual_decorator


def returns_plottable(identifier: str):
    def actual_decorator(func: Callable[..., Callable]):
        @wraps(func)
        def transformed_function(*args, **kwargs):
            return plottable(identifier)(func(*args, **kwargs))
        return transformed_function
    return actual_decorator


def add_callback(identifier: str, callback: Callable):
    _function_registry.add_callback(identifier, callback)
