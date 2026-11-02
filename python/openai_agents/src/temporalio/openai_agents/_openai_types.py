"""Decode the ``Iterable`` fields of the openai item types as lists.

Pydantic decodes an ``Iterable[X]`` field as a one-shot iterator, which cannot be copied, and in
JSON mode it picks the ``Iterable`` side of ``str | Iterable[X]`` for a string
(pydantic/pydantic#11219). Importing this module swaps those annotations for ``list[X]``, so every
class built afterwards decodes carried items as lists and strings. Tool definitions keep their
types, because other ``openai`` requests share them.
"""

import collections.abc
import functools
import operator
import types
import typing

import typing_extensions
from agents import TResponseInputItem
from openai.types.responses import ToolParam


def _as_list(hint: typing.Any) -> typing.Any:
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin is collections.abc.Iterable:
        return types.GenericAlias(list, (_as_list(args[0]),))
    if origin is None or not args:
        return hint
    if origin is typing_extensions.Annotated:
        inner = _as_list(args[0])
        return (
            hint
            if inner is args[0]
            else typing_extensions.Annotated[(inner, *args[1:])]
        )
    new = tuple(_as_list(arg) for arg in args)
    if new == args:
        return hint
    if isinstance(hint, types.UnionType):
        return functools.reduce(operator.or_, new)
    return hint.copy_with(new)


def _swap(hint: typing.Any, seen: set[int]) -> None:
    if typing_extensions.is_typeddict(hint):
        if id(hint) in seen:
            return
        seen.add(id(hint))
        hints = typing_extensions.get_type_hints(hint, include_extras=True)
        for name, field in hints.items():
            hint.__annotations__[name] = _as_list(field)
            _swap(field, seen)
        return
    for arg in typing.get_args(hint):
        _swap(arg, seen)


# Items can hold tool definitions; start with them marked as seen.
_swap(TResponseInputItem, {id(tool) for tool in typing.get_args(ToolParam)})
