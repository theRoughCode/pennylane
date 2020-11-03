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
"""Unit tests for the mxnet interface"""
import pytest

# mx = pytest.importorskip("mxnet", minversion="1.6")
import mxnet as mx
import numpy as np

import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.tape.interfaces.mx import MXInterface


class TestMXQuantumTape:
    """Test the MXNet interface applied to a tape"""

    def test_interface_construction(self):
        """Test that the interface is correctly applied"""
        with MXInterface.apply(QuantumTape()) as tape:
            qml.RX(0.5, wires=0)
            qml.expval(qml.PauliX(0))

        assert tape.interface == "mx"
        assert isinstance(tape, MXInterface)
        assert tape.__bare__ == QuantumTape

    def test_repeated_interface_construction(self):
        """Test that the interface is correctly applied multiple times"""
        with MXInterface.apply(QuantumTape()) as tape:
            qml.RX(0.5, wires=0)
            qml.expval(qml.PauliX(0))

        assert tape.interface == "mx"
        assert isinstance(tape, MXInterface)
        assert tape.__bare__ == QuantumTape

        MXInterface.apply(tape, dtype=np.float64)
        assert tape.interface == "mx"
        assert isinstance(tape, MXInterface)
        assert tape.__bare__ == QuantumTape
        assert tape.dtype is np.float64

    def test_get_parameters(self):
        """Test that the get parameters function correctly sets and returns the
        trainable parameters"""
        a = mx.nd.array([0.1])
        b = mx.nd.array([0.2])
        c = mx.nd.array([0.3])
        d = 0.4
        a.attach_grad()
        c.attach_grad()

        with MXInterface.apply(QuantumTape()) as tape:
            qml.Rot(a, b, c, wires=0)
            qml.RX(d, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliX(0))

        # NOTE: We treat all NDArray params as trainable since there is no way to
        # figure out which ones are trainable with the MXNet API
        assert tape.trainable_params == {0, 1, 2}
        assert np.all(tape.get_parameters() == [a, b, c])

    def test_execution(self):
        """Test execution"""
        a = mx.nd.array([0.1])
        a.attach_grad()
        dev = qml.device("default.qubit", wires=1)

        with MXInterface.apply(QuantumTape()) as tape:
            qml.RY(a, wires=0)
            qml.RX(mx.nd.array([0.2]), wires=0)
            qml.expval(qml.PauliZ(0))

        assert tape.trainable_params == {0, 1}
        res = tape.execute(dev)

        assert isinstance(res, mx.nd.NDArray)
        assert res.shape == (1,)

    def test_jacobian(self, mocker, tol):
        """Test jacobian calculation"""
        spy = mocker.spy(QuantumTape, "jacobian")

        a_val = 0.1
        b_val = 0.2

        a = mx.nd.array([a_val])
        a.attach_grad()
        b = mx.nd.array([b_val])
        b.attach_grad()

        dev = qml.device("default.qubit", wires=2)

        with mx.autograd.record():
            with MXInterface.apply(QuantumTape()) as tape:
                qml.RZ(mx.nd.array(0.543), wires=0)
                qml.RY(a, wires=0)
                qml.RX(b, wires=1)
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.expval(qml.PauliY(1))

            assert tape.trainable_params == {0, 1, 2}
            res = tape.execute(dev)
            res.backward()

        assert isinstance(res, mx.nd.NDArray)
        assert res.shape == (2,)

        expected = [np.cos(a_val), -np.cos(a_val) * np.sin(b_val)]
        assert np.allclose(res.asnumpy(), expected, atol=tol, rtol=0)

        expected = [
            -np.sin(a_val) + np.sin(a_val) * np.sin(b_val),
            -np.cos(a_val) * np.cos(b_val),
        ]
        print(a.grad, "HIIIIIIIIIIIIII")
        print(a.grad.asscalar())
        assert np.allclose(a.grad.asscalar(), expected[0], atol=tol, rtol=0)
        assert np.allclose(b.grad.asscalar(), expected[1], atol=tol, rtol=0)

        spy.assert_called()

    def test_jacobian_options(self, mocker, tol):
        """Test jacobian calculation"""
        spy = mocker.spy(QuantumTape, "jacobian")

        a_val = 0.1
        b_val = 0.2

        a = mx.nd.array([a_val, 0.2])
        a.attach_grad()

        dev = qml.device("default.qubit", wires=2)

        with mx.autograd.record():
            with MXInterface.apply(QuantumTape()) as tape:
                qml.RZ(mx.nd.array(0.543), wires=0)
                qml.RY(a[0], wires=0)
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.expval(qml.PauliY(1))

            res = tape.execute(dev)
            res.backward()


#     def test_jacobian_dtype(self, tol):
#         """Test calculating the jacobian with a different datatype"""
#         a_val = 0.1
#         b_val = 0.2

#         a = mx.nd.array(a_val, requires_grad=True, dtype=np.float32)
#         b = mx.nd.array(b_val, requires_grad=True, dtype=np.float32)

#         dev = qml.device("default.qubit", wires=2)

#         with MXInterface.apply(QuantumTape(), dtype=np.float32) as tape:
#             qml.RY(a, wires=0)
#             qml.RX(b, wires=1)
#             qml.CNOT(wires=[0, 1])
#             qml.expval(qml.PauliZ(0))
#             qml.expval(qml.PauliY(1))

#         assert tape.trainable_params == {0, 1}
#         res = tape.execute(dev)

#         assert isinstance(res, mx.nd.array)
#         assert res.shape == (2,)
#         assert res.dtype is np.float32

#         loss = mx.sum(res)
#         loss.backward()
#         assert a.grad.dtype is np.float32
#         assert b.grad.dtype is np.float32

#     def test_reusing_quantum_tape(self, tol):
#         """Test re-using a quantum tape by passing new parameters"""
#         a = mx.nd.array(0.1, requires_grad=True)
#         b = mx.nd.array([0.2], requires_grad=True)

#         dev = qml.device("default.qubit", wires=2)

#         with MXInterface.apply(QuantumTape()) as tape:
#             qml.RY(a, wires=0)
#             qml.RX(b, wires=1)
#             qml.CNOT(wires=[0, 1])
#             qml.expval(qml.PauliZ(0))
#             qml.expval(qml.PauliY(1))

#         assert tape.trainable_params == {0, 1}

#         loss = mx.sum(tape.execute(dev))
#         loss.backward()

#         a_val = 0.54
#         b_val = 0.8
#         a = mx.nd.array(a_val, requires_grad=True)
#         b = mx.nd.array(b_val, requires_grad=True)
#         res2 = tape.execute(dev, params=[2 * a, b])

#         expected = [np.cos(2 * a_val), -np.cos(2 * a_val) * np.sin(b_val)]
#         assert np.allclose(res2.detach().numpy(), expected, atol=tol, rtol=0)

#         loss = mx.sum(res2)
#         loss.backward()

#         expected = [
#             -2 * np.sin(2 * a_val) + 2 * np.sin(2 * a_val) * np.sin(b_val),
#             -np.cos(2 * a_val) * np.cos(b_val),
#         ]

#         assert np.allclose(a.grad, expected[0], atol=tol, rtol=0)
#         assert np.allclose(b.grad, expected[1], atol=tol, rtol=0)

#     def test_classical_processing(self, tol):
#         """Test classical processing within the quantum tape"""
#         p_val = [0.1, 0.2]
#         params = mx.nd.array(p_val, requires_grad=True)

#         dev = qml.device("default.qubit", wires=1)

#         with MXInterface.apply(QuantumTape()) as tape:
#             qml.RY(params[0] * params[1], wires=0)
#             qml.RZ(0.2, wires=0)
#             qml.RX(params[1] + params[1] ** 2 + mx.sin(params[0]), wires=0)
#             qml.expval(qml.PauliZ(0))

#         assert tape.trainable_params == {0, 2}

#         tape_params = [i.detach().numpy() for i in tape.get_parameters()]
#         assert np.allclose(
#             tape_params,
#             [p_val[0] * p_val[1], p_val[1] + p_val[1] ** 2 + np.sin(p_val[0])],
#             atol=tol,
#             rtol=0,
#         )

#         res = tape.execute(dev)
#         res.backward()

#         assert isinstance(params.grad, mx.nd.array)
#         assert params.shape == (2,)

#     def test_no_trainable_parameters(self, tol):
#         """Test evaluation and Jacobian if there are no trainable parameters"""
#         dev = qml.device("default.qubit", wires=2)

#         with MXInterface.apply(QuantumTape()) as tape:
#             qml.RY(0.2, wires=0)
#             qml.RX(mx.nd.array(0.1), wires=0)
#             qml.CNOT(wires=[0, 1])
#             qml.expval(qml.PauliZ(0))
#             qml.expval(qml.PauliZ(1))

#         assert tape.trainable_params == set()

#         res = tape.execute(dev)

#         assert res.shape == (2,)
#         assert isinstance(res, mx.nd.array)

#         with pytest.raises(
#             RuntimeError,
#             match="element 0 of tensors does not require grad and does not have a grad_fn",
#         ):
#             res.backward()

#     @pytest.mark.parametrize(
#         "U", [mx.nd.array([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]])]
#     )
#     def test_matrix_parameter(self, U, tol):
#         """Test that the mx interface works correctly
#         with a matrix parameter"""
#         a_val = 0.1
#         a = mx.nd.array(a_val, requires_grad=True)

#         dev = qml.device("default.qubit", wires=2)

#         with MXInterface.apply(QuantumTape()) as tape:
#             qml.QubitUnitary(U, wires=0)
#             qml.RY(a, wires=0)
#             qml.expval(qml.PauliZ(0))

#         assert tape.trainable_params == {1}
#         res = tape.execute(dev)

#         assert np.allclose(res.detach().numpy(), -np.cos(a_val), atol=tol, rtol=0)

#         res.backward()
#         assert np.allclose(a.grad, np.sin(a_val), atol=tol, rtol=0)

#     def test_differentiable_expand(self, mocker, tol):
#         """Test that operation and nested tapes expansion
#         is differentiable"""
#         mock = mocker.patch.object(qml.operation.Operation, "do_check_domain", False)

#         class U3(qml.U3):
#             def expand(self):
#                 tape = QuantumTape()
#                 theta, phi, lam = self.data
#                 wires = self.wires
#                 tape._ops += [
#                     qml.Rot(lam, theta, -lam, wires=wires),
#                     qml.PhaseShift(phi + lam, wires=wires),
#                 ]
#                 return tape

#         tape = QuantumTape()

#         dev = qml.device("default.qubit", wires=1)
#         a = np.array(0.1)
#         p_val = [0.1, 0.2, 0.3]
#         p = mx.nd.array(p_val, requires_grad=True)

#         with tape:
#             qml.RX(a, wires=0)
#             U3(p[0], p[1], p[2], wires=0)
#             qml.expval(qml.PauliX(0))

#         tape = MXInterface.apply(tape.expand())

#         assert tape.trainable_params == {1, 2, 3, 4}
#         assert [i.name for i in tape.operations] == ["RX", "Rot", "PhaseShift"]

#         tape_params = [i.detach().numpy() for i in tape.get_parameters()]
#         assert np.allclose(
#             tape_params,
#             [p_val[2], p_val[0], -p_val[2], p_val[1] + p_val[2]],
#             atol=tol,
#             rtol=0,
#         )

#         res = tape.execute(device=dev)

#         expected = np.cos(a) * np.cos(p_val[1]) * np.sin(p_val[0]) + np.sin(a) * (
#             np.cos(p_val[2]) * np.sin(p_val[1])
#             + np.cos(p_val[0]) * np.cos(p_val[1]) * np.sin(p_val[2])
#         )
#         assert np.allclose(res.detach().numpy(), expected, atol=tol, rtol=0)

#         res.backward()
#         expected = np.array(
#             [
#                 np.cos(p_val[1])
#                 * (
#                     np.cos(a) * np.cos(p_val[0])
#                     - np.sin(a) * np.sin(p_val[0]) * np.sin(p_val[2])
#                 ),
#                 np.cos(p_val[1]) * np.cos(p_val[2]) * np.sin(a)
#                 - np.sin(p_val[1])
#                 * (
#                     np.cos(a) * np.sin(p_val[0])
#                     + np.cos(p_val[0]) * np.sin(a) * np.sin(p_val[2])
#                 ),
#                 np.sin(a)
#                 * (
#                     np.cos(p_val[0]) * np.cos(p_val[1]) * np.cos(p_val[2])
#                     - np.sin(p_val[1]) * np.sin(p_val[2])
#                 ),
#             ]
#         )
#         assert np.allclose(p.grad, expected, atol=tol, rtol=0)

#     def test_probability_differentiation(self, tol):
#         """Tests correct output shape and evaluation for a tape
#         with multiple prob outputs"""

#         dev = qml.device("default.qubit", wires=2)
#         x_val = 0.543
#         y_val = -0.654
#         x = mx.nd.array(x_val, requires_grad=True)
#         y = mx.nd.array(y_val, requires_grad=True)

#         with MXInterface.apply(QuantumTape()) as tape:
#             qml.RX(x, wires=[0])
#             qml.RY(y, wires=[1])
#             qml.CNOT(wires=[0, 1])
#             qml.probs(wires=[0])
#             qml.probs(wires=[1])

#         res = tape.execute(dev)

#         expected = np.array(
#             [
#                 [np.cos(x_val / 2) ** 2, np.sin(x_val / 2) ** 2],
#                 [
#                     (1 + np.cos(x_val) * np.cos(y_val)) / 2,
#                     (1 - np.cos(x_val) * np.cos(y_val)) / 2,
#                 ],
#             ]
#         )
#         assert np.allclose(res.detach().numpy(), expected, atol=tol, rtol=0)

#         loss = mx.sum(res)
#         loss.backward()
#         expected = np.array(
#             [
#                 -np.sin(x_val) / 2
#                 + np.sin(x_val) / 2
#                 - np.sin(x_val) * np.cos(y_val) / 2
#                 + np.cos(y_val) * np.sin(x_val) / 2,
#                 -np.cos(x_val) * np.sin(y_val) / 2 + np.cos(x_val) * np.sin(y_val) / 2,
#             ]
#         )
#         assert np.allclose(x.grad, expected[0], atol=tol, rtol=0)
#         assert np.allclose(y.grad, expected[1], atol=tol, rtol=0)

#     def test_ragged_differentiation(self, tol):
#         """Tests correct output shape and evaluation for a tape
#         with prob and expval outputs"""
#         dev = qml.device("default.qubit", wires=2)
#         x_val = 0.543
#         y_val = -0.654
#         x = mx.nd.array(x_val, requires_grad=True)
#         y = mx.nd.array(y_val, requires_grad=True)

#         with MXInterface.apply(QuantumTape()) as tape:
#             qml.RX(x, wires=[0])
#             qml.RY(y, wires=[1])
#             qml.CNOT(wires=[0, 1])
#             qml.expval(qml.PauliZ(0))
#             qml.probs(wires=[1])

#         res = tape.execute(dev)

#         expected = np.array(
#             [
#                 np.cos(x_val),
#                 (1 + np.cos(x_val) * np.cos(y_val)) / 2,
#                 (1 - np.cos(x_val) * np.cos(y_val)) / 2,
#             ]
#         )
#         assert np.allclose(res.detach().numpy(), expected, atol=tol, rtol=0)

#         loss = mx.sum(res)
#         loss.backward()
#         expected = np.array(
#             [
#                 -np.sin(x_val)
#                 + -np.sin(x_val) * np.cos(y_val) / 2
#                 + np.cos(y_val) * np.sin(x_val) / 2,
#                 -np.cos(x_val) * np.sin(y_val) / 2 + np.cos(x_val) * np.sin(y_val) / 2,
#             ]
#         )
#         assert np.allclose(x.grad, expected[0], atol=tol, rtol=0)
#         assert np.allclose(y.grad, expected[1], atol=tol, rtol=0)

#     def test_sampling(self):
#         """Test sampling works as expected"""
#         dev = qml.device("default.qubit", wires=2, shots=10)

#         with MXInterface.apply(QuantumTape()) as tape:
#             qml.Hadamard(wires=[0])
#             qml.CNOT(wires=[0, 1])
#             qml.sample(qml.PauliZ(0))
#             qml.sample(qml.PauliX(1))

#         res = tape.execute(dev)

#         assert res.shape == (2, 10)
#         assert isinstance(res, mx.nd.array)

#     def test_complex_min_version(self, monkeypatch):
#         """Test if an error is raised when a version of mx before 1.6.0 is used as the dtype
#         in the apply() method"""

#         with monkeypatch.context() as m:
#             m.setattr(qml.tape.interfaces.mx, "COMPLEX_SUPPORT", False)
#             with pytest.raises(
#                 qml.QuantumFunctionError, match="Version 1.6.0 or above of Pymx"
#             ):
#                 MXInterface.apply(QuantumTape(), dtype=mx.complex128)
