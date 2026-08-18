"""标量自动微分引擎的练习框架。"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence


Number = int | float


def numerical_derivative(
    fn: Callable[[float], float],
    x: float,
    h: float = 1e-6,
) -> float:
    delta = fn(x+h) - fn(x)
    return (delta / h)


class Value:
    """计算图中的一个标量节点。"""

    def __init__(
        self,
        data: Number,
        _children: tuple[Value, ...] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
       return f"data: {self.data} \n label: {self.label} \n grad: {self.grad} \n"

    @staticmethod
    def _coerce(other: Value | Number) -> Value:
        """把普通数字包装成 Value；这是框架提供的辅助函数。"""
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other: Value | Number) -> Value:
        other = self._coerce(other)
        value = Value(
            self.data + other.data,
            (self, other),
            "+"
        )

        def _backward():
            self.grad += value.grad
            other.grad += value.grad

        value._backward = _backward

        return value

    def __mul__(self, other: Value | Number) -> Value:
        other = self._coerce(other)
        value = Value(
            self.data * other.data,
            (self, other),
            "*"
        )

        def _backward():
            self.grad += value.grad * other.data
            other.grad += value.grad * self.data
        
        value._backward = _backward
        
        return value

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        value = Value(
            t,
            (self,),
            "tanh"
        )

        def _backward():
            self.grad += value.grad * (1 - t**2)

        value._backward = _backward
        return value
        

    def backward(self) -> None:
        topo = []
        visited = set()

        def build(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for ch in v._prev:
                    build(ch)
                topo.append(v)
    
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __radd__(self, other: Value | Number) -> Value:
        return self.__add__(other)

    def __rmul__(self, other: Value | Number) -> Value:
        return self.__mul__(other)

    def __neg__(self) -> Value:
        t = - self.data
        value = Value(t, (self,), "neg")
        def _backward():
            self.grad -= value.grad

        value._backward = _backward
        return value

    def __sub__(self, other: Value | Number) -> Value:
        other = self._coerce(other)
        t = self.data - other.data
        value = Value(t, (self,), "neg")
        def _backward():
            self.grad += value.grad
            other.grad -= value.grad

        value._backward = _backward
        return value        

    def __rsub__(self, other: Value | Number) -> Value:
        return Value(other).__sub__(self)

    def __pow__(self, e: Number) -> Value:
        t = self.data ** e
        value = Value(t, (self,), "pow")
        def _backward():
            self.grad += value.grad * e * (self.data ** (e - 1))
        
        value._backward = _backward
        return value        

    def __truediv__(self, other: Value | Number) -> Value:
        other = self._coerce(other)
        t = self.data / other.data
        value = Value(t, (self,), "div")
        def _backward():
            self.grad += value.grad * (1.0 / other.data)
            other.grad -= value.grad * (self.data / (other.data ** 2))  

        value._backward = _backward
        return value       

    def __rtruediv__(self, other: Value | Number) -> Value:
        return Value(other).__truediv__(self)

    def exp(self) -> Value:
        t = math.exp(self.data)
        value = Value(t, (self,), "exp")
        def _backward():
            self.grad += value.grad * t
        
        value._backward = _backward
        return value


def manual_backprop_ab_plus_c(
    a: float,
    b: float,
    c: float,
) -> dict[str, float]:
    aa, bb, cc = Value(a, label="a"), Value(b,label="b"), Value(c,label="c")
    product = aa * bb
    out = product + cc
    out.grad = 1.0
    out._backward()
    product._backward()

    return {"out": out.grad, "product": product.grad, "a": aa.grad, "b": bb.grad, "c": cc.grad}


def gradient_check(
    fn: Callable[..., Value],
    inputs: Sequence[float],
    eps: float = 1e-6,
) -> list[tuple[float, float]]:
    _inputs = []
    for i in inputs:
        _inputs.append(Value._coerce(i))
    ret = []
    v = fn(*_inputs)
    v.backward()
    for i, x in enumerate(_inputs):
        iinput = _inputs.copy()
        xx = Value(x.data + eps)
        iinput[i] = xx
        vv = fn(*iinput)
        numericalGrad = ((vv-v)/eps).data
        ret.append((numericalGrad, x.grad))

    return ret
