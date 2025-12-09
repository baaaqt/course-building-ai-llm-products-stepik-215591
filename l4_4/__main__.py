import math
from pprint import pprint
from typing import TypedDict, NotRequired
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence,
)


class QuadraticEquationDataFirst(TypedDict):
    a: float
    b: float
    c: float
    discriminant: NotRequired[float]


class QuadraticEquationDataSecond(TypedDict):
    a: float
    b: float
    c: float
    discriminant: float


discriminant_runnable = RunnablePassthrough.assign(
    discriminant=RunnableLambda(lambda data: data["b"] ** 2 - 4 * data["a"] * data["c"])
)


one_root_solution = RunnableLambda(lambda nums: -nums["b"] / (2 * nums["a"]))


two_roots_x1 = RunnableLambda(
    lambda nums: (-nums["b"] + math.sqrt(nums["discriminant"])) / (2 * nums["a"])
)

two_roots_x2 = RunnableLambda(
    lambda nums: (-nums["b"] - math.sqrt(nums["discriminant"])) / (2 * nums["a"])
)

two_roots_solution = RunnableParallel(x1=two_roots_x1, x2=two_roots_x2)


def roots_runable(
    nums: QuadraticEquationDataSecond,
) -> Runnable | None:
    if nums["discriminant"] > 0.0:
        return two_roots_solution
    elif nums["discriminant"] == 0.0:
        return one_root_solution
    return None


roots = RunnableLambda(roots_runable)


pipeline = RunnableSequence(discriminant_runnable, roots)

pprint(pipeline.get_graph().draw_ascii())
print("\n\n")

result = pipeline.invoke({"a": 1, "b": -5, "c": 6})
assert set((result["x1"], result["x2"])) == {2.0, 3.0}
assert pipeline.invoke({"a": 1, "b": 2, "c": 1}) == -1.0
assert pipeline.invoke({"a": 1, "b": 0, "c": 0}) == 0.0
assert pipeline.invoke({"a": 1, "b": 0, "c": 1}) is None

print("All tests passed")
