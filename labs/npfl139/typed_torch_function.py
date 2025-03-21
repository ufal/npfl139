# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import torch


def typed_torch_function(device, *types, via_np=False):
    """Typed Torch function decorator.

    The positional input arguments are converted to torch Tensors of the given
    types and on the given device; for NumPy arrays on the same device,
    the conversion should not copy the data.

    The torch Tensors generated by the wrapped function are converted back
    to Numpy arrays before returning (while keeping original tuples, lists,
    and dictionaries).
    """
    def check_typed_torch_function(wrapped, args):
        if len(types) != len(args):
            while hasattr(wrapped, "__wrapped__"):
                wrapped = wrapped.__wrapped__
            raise AssertionError("The typed_torch_function decorator for {} expected {} arguments, but got {}".format(
                wrapped, len(types), len(args)))

    def structural_map(value):
        if isinstance(value, torch.Tensor):
            return value.numpy(force=True)
        if isinstance(value, tuple):
            return tuple(structural_map(element) for element in value)
        if isinstance(value, list):
            return [structural_map(element) for element in value]
        if isinstance(value, dict):
            return {key: structural_map(element) for key, element in value.items()}
        return value

    class TypedTorchFunctionWrapper:
        def __init__(self, func):
            self.__wrapped__ = func

        def __call__(self, *args, **kwargs):
            check_typed_torch_function(self.__wrapped__, args)
            return structural_map(self.__wrapped__(
                *[torch.as_tensor(np.asarray(arg) if via_np else arg, dtype=typ, device=device)
                  for arg, typ in zip(args, types)], **kwargs))

        def __get__(self, instance, cls):
            return TypedTorchFunctionWrapper(self.__wrapped__.__get__(instance, cls))

    return TypedTorchFunctionWrapper
