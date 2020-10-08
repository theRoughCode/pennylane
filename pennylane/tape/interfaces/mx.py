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
This module contains the mixin interface class for creating differentiable quantum tapes with
PyTorch.
"""
from mxnet import autograd
import numpy as np
import mxnet as mx

from pennylane.tape.queuing import AnnotatedQueue


class _MXInterface(mx.autograd.Function):
    @staticmethod
    def convert_to_numpy(tensors):
        """Converts any MXNet NDArray in a sequence to NumPy arrays.

        Args:
            tensors (Sequence[Any, mx.nd.NDArray]): input sequence

        Returns:
            list[Any, array]: list with all tensors converted to NumPy arrays
        """
        return [i.asnumpy() if isinstance(i, mx.nd.NDArray) else i for i in tensors]

    def forward(self, input_kwargs, *input_):
        """Implements the forward pass QNode evaluation"""
        # detach all input tensors, convert to NumPy array
        self.args = self.convert_to_numpy(input_)
        self.kwargs = input_kwargs
        self.save_for_backward(*input_)

        tape = self.kwargs["tape"]
        device = self.kwargs["device"]

        # unwrap constant parameters
        self.all_params = tape.get_parameters(trainable_only=False)
        self.all_params_unwrapped = self.convert_to_numpy(self.all_params)

        # evaluate the tape
        tape.set_parameters(self.all_params_unwrapped, trainable_only=False)
        res = tape.execute_device(self.args, device)
        tape.set_parameters(self.all_params, trainable_only=False)

        # if any input tensor uses the GPU, the output should as well
        for i in input_:
            if isinstance(i, mx.nd.NDArray):
                context = i.context
                if context.device_type == "gpu":
                    return mx.nd.array(res, ctx=context, dtype=tape.dtype)

        if res.dtype == np.dtype("object"):
            res = np.hstack(res)

        return mx.nd.array(res, dtype=tape.dtype)

    def backward(self, dy):
        """Implements the backwards pass QNode vector-Jacobian product"""
        tape = self.kwargs["tape"]
        device = self.kwargs["device"]

        tape.set_parameters(self.all_params_unwrapped, trainable_only=False)
        jacobian = tape.jacobian(device, params=self.args, **tape.jacobian_options)
        tape.set_parameters(self.all_params, trainable_only=False)

        jacobian = mx.nd.array(jacobian, dtype=dy.dtype)

        # Calculate the vector-Jacobian matrix product, and unstack the output.
        grad_input = dy.reshape(1, -1) @ jacobian
        flattened_grad = grad_input.reshape(-1)
        grad_input_list = flattened_grad.split(flattened_grad.shape[0], axis=0)
        grad_input = []

        # match the type and device of the input tensors
        (inputs,) = self.saved_tensors
        for i, j in zip(grad_input_list, inputs):
            res = mx.nd.array(i, dtype=tape.dtype)
            context = j.context
            if context.device_type == "gpu":
                res = mx.nd.array(res, ctx=context, dtype=tape.dtype)
            grad_input.append(res)

        return (None,) + tuple(grad_input)


class sigmoid(mx.autograd.Function):
    def forward(self, _, x):
        print("hello", x)
        y = 1 / (1 + mx.nd.exp(-x))
        self.save_for_backward(y)
        print("bye")
        return y

    def backward(self, dy):
        print("asdsad")
        (y,) = self.saved_tensors
        return (None, dy * y * (1 - y))


class MXInterface(AnnotatedQueue):
    """Mixin class for applying an MXNet interface to a :class:`~.QuantumTape`.

    MXNet-compatible quantum tape classes can be created via subclassing:

    .. code-block:: python

        class MyMXQuantumTape(MXInterface, QuantumTape):

    Alternatively, the MXNet interface can be dynamically applied to existing
    quantum tapes via the :meth:`~.apply` class method. This modifies the
    tape **in place**.

    Once created, the MXNet interface can be used to perform quantum-classical
    differentiable programming.

    **Example**

    Once a MXNet quantum tape has been created, it can be evaluated and differentiated:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=1)
        p = mx.nd.array([0.1, 0.2, 0.3])

        with MXInterface.apply(QuantumTape()) as qtape:
            qml.Rot(p[0], p[1] ** 2 + p[0] * p[2], p[1] * mx.nd.sin(p[2]), wires=0)
            expval(qml.PauliX(0))

        result = qtape.execute(dev)

    >>> print(result)
    [0.06982073]
    <NDArray 1 @cpu(0)>
    >>> result.backward()
    >>> print(p.grad)
    [0.29874274 0.39710271 0.09958091]
    <NDArray 3 @cpu(0)>

    The MXNet interface defaults to ``numpy.float32`` output. This can be modified by
    providing the ``dtype`` argument when applying the interface:

    >>> p = mx.nd.array([0.1, 0.2, 0.3], dtype=numpy.float64)
    >>> with MXInterface.apply(QuantumTape()) as qtape:
    ...     qml.Rot(p[0], p[1] ** 2 + p[0] * p[2], p[1] *  mx.nd.sin(p[2]), wires=0)
    ...     expval(qml.PauliX(0))
    >>> result = qtape.execute(dev)
    >>> print(result)
    [0.06982072]
    <NDArray 1 @cpu(0)>
    >>> print(result.dtype)
    numpy.float64
    >>> result.backward()
    >>> print(p.grad)
    [0.29874274 0.39710271 0.09958091]
    <NDArray 3 @cpu(0)>
    >>> print(p.grad.dtype)
    numpy.float64
    """

    dtype = np.float64
    f = sigmoid()

    @property
    def interface(self):  # pylint: disable=missing-function-docstring
        return "mx"

    def _update_trainable_params(self):
        params = self.get_parameters(trainable_only=False)

        trainable_params = set()

        for idx, p in enumerate(params):
            if isinstance(p, mx.nd.NDArray) and p.grad is None:
                trainable_params.add(idx)

        self.trainable_params = trainable_params
        return params

    def _execute(self, params, **kwargs):
        kwargs["tape"] = self
        x = params[-1]
        res = self.f("a", x)
        return res

    @classmethod
    def apply(cls, tape, dtype=np.float64):
        """Apply the MXNet interface to an existing tape in-place.

        Args:
            tape (.QuantumTape): a quantum tape to apply the MXNet interface to
            dtype (np.dtype): the dtype that the returned quantum tape should
                output

        **Example**

        >>> with QuantumTape() as tape:
        ...     qml.RX(0.5, wires=0)
        ...     expval(qml.PauliZ(0))
        >>> MXInterface.apply(tape)
        >>> tape
        <MXQuantumTape: wires=[0], params=0>
        """
        tape_class = getattr(tape, "__bare__", tape.__class__)
        tape.__bare__ = tape_class
        tape.__class__ = type("MXQuantumTape", (cls, tape_class), {"dtype": dtype})
        tape._update_trainable_params()
        return tape
