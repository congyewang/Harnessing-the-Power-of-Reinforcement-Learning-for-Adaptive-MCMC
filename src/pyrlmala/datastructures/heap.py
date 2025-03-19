import heapq
from typing import Callable, Generic, List, Optional, TypeVar

T = TypeVar("T")


class DynamicTopK(Generic[T]):
    """
    A class to maintain the top k elements in a stream of data.
    It uses a min-heap to keep track of the top k elements.
    """

    def __init__(self, k: int, key: Optional[Callable[[T], float]] = None):
        """
        Initialize the DynamicTopK class.

        Args:
            k (int): The number of top elements to maintain.
            key (Optional[Callable[[T], float]], optional): A function to extract a comparison key from each element. Defaults to None.
        """
        self.k: int = k
        self.heap: List[T] = []
        self.key = key

    def add(self, val: T) -> None:
        """
        Add an element to the top k elements.

        Args:
            val (T): The element to be added to the top k.
        """
        val_key = self.key(val) if self.key else val
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, (val_key, val))
        else:
            if val_key > self.heap[0][0]:
                heapq.heappushpop(self.heap, (val_key, val))

    def topk(self) -> List[T]:
        """
        Get the top k elements.

        Returns:
            List[T]: The top k elements in descending order.
        """
        return [item[1] for item in sorted(self.heap, key=lambda x: x[0], reverse=True)]
