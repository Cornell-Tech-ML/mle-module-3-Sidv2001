from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Dict

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    res = list(vals)
    res[arg] += epsilon
    forward = f(*res)
    res[arg] -= 2 * epsilon
    backward = f(*res)
    derr = (forward - backward) / (2 * epsilon)
    return derr


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    # child_num: Dict[int, int] = {}
    # stack = [variable]
    # while len(stack) > 0:
    #     scal = stack.pop()
    #     if scal.unique_id in child_num:
    #         child_num[scal.unique_id] += 1
    #     else:
    #         child_num[scal.unique_id] = 1
    #         stack.extend(scal.parents)

    # no_dependency = [variable]
    # fin = []
    # while len(no_dependency) > 0:
    #     scal = no_dependency.pop()
    #     if not scal.is_constant():
    #         fin.append(scal)
    #     for parent in scal.parents:
    #         if child_num[parent.unique_id] == 1:
    #             no_dependency.append(parent)
    #         else:
    #             child_num[parent.unique_id] -= 1
    # return fin
    order: List[Variable] = []
    seen = set()

    def visit(var: Variable) -> None:
        if var.unique_id in seen or var.is_constant():
            return
        if not var.is_leaf():
            for m in var.parents:
                if not m.is_constant():
                    visit(m)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    top_sort_vars = topological_sort(variable)
    total_vars = len(list(top_sort_vars))
    res = [0] * total_vars
    derrs = dict((key.unique_id, value) for key, value in zip(top_sort_vars, res))

    derrs[variable.unique_id] = deriv
    for var in top_sort_vars:
        if not var.is_leaf():
            chained = var.chain_rule(derrs[var.unique_id])
            for res_var, der in chained:
                if res_var.is_leaf():
                    res_var.accumulate_derivative(der)
                elif res_var.is_constant():
                    pass
                else:
                    derrs[res_var.unique_id] += der


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
