# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains the :class:`Wires` class.
"""
from collections import Set, Sequence


class WireError(Exception):
    """Exception raised by a :class:`~.pennylane.wires.Wire` when it is unable to process wire objects.
    """


class Wires(Sequence, Set):

    def __init__(self, wires):
        """
        A bookkeeping class for wires, which are ordered collections of unique non-negative integers that
        represent the index of a wire.

        Args:
             wires (iterable): Ordered collection of unique wire indices. Takes common types of iterables,
                such as lists, tuples and numpy arrays. The element of the iterable must be a
                non-negative integer. If elements are floats, they are internally converted to integers,
                throwing an error if the rounding error exceeds TOLERANCE.
        """

        if len(set(wires)) != len(wires):
            raise WireError("Each wire must be represented by a unique index; got {}.".format(wires))

        if wires is not None:
            self._wires = list(wires)

    def __getitem__(self, idx):
        return self._wires[idx]

    def __len__(self):
        return len(self._wires)
