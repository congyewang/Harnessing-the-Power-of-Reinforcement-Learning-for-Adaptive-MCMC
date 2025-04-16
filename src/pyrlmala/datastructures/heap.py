import heapq
from itertools import count
from typing import Callable, Generic, List, Optional, TypeVar

T = TypeVar("T")


class DynamicTopK(Generic[T]):
    """
    A class to maintain the top k elements in a stream of data.
    It uses a min-heap to keep track of the top k elements.
    """

    def __init__(self, k: int, key: Optional[Callable[[T], float]] = None) -> None:
        """
        Initialize the DynamicTopK class.

        Args:
            k (int): The number of top elements to maintain. Must be greater than 0.
            key (Optional[Callable[[T], float]], optional): A function to extract a comparison key from each element. Defaults to None.
        """
        if k <= 0:
            raise ValueError("k must be greater than 0")
        self.k: int = k
        self.heap: List[T] = []
        self.key = key

        self.counter = count()

    def add(self, val: T) -> None:
        """
        Add an element to the top k elements.

        Args:
            val (T): The element to be added to the top k.
        """
        count_id = next(self.counter)

        val_key = self.key(val) if self.key else val
        item = (val_key, count_id, val)

        if len(self.heap) < self.k:
            heapq.heappush(self.heap, item)
        else:
            if val_key > self.heap[0][0]:
                heapq.heappushpop(self.heap, item)

    def topk(self) -> List[T]:
        """
        Get the top k elements.

        Returns:
            List[T]: The top k elements in descending order.
        """
        return [item[2] for item in sorted(self.heap, key=lambda x: x[0], reverse=True)]
