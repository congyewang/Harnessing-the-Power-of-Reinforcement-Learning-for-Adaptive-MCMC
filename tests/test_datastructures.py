import pytest
from pyrlmala.datastructures.heap import DynamicTopK


class TestDynamicTopK:
    def test_init_valid(self):
        heap = DynamicTopK(k=3)
        assert heap.k == 3
        assert len(heap.heap) == 0
        assert heap.key is None

    def test_init_with_key_function(self):
        key_func = lambda x: -x
        heap = DynamicTopK(k=3, key=key_func)
        assert heap.k == 3
        assert heap.key == key_func

    def test_init_invalid_k(self):
        with pytest.raises(ValueError):
            DynamicTopK(k=0)
        with pytest.raises(ValueError):
            DynamicTopK(k=-1)

    def test_add_below_capacity(self):
        heap = DynamicTopK(k=3)
        heap.add(1)
        heap.add(2)
        assert len(heap.heap) == 2
        assert set(heap.topk()) == {1, 2}

    def test_add_at_capacity(self):
        heap = DynamicTopK(k=3)
        heap.add(1)
        heap.add(2)
        heap.add(3)
        heap.add(4)
        assert len(heap.heap) == 3
        assert set(heap.topk()) == {2, 3, 4}

    def test_add_with_key_function(self):
        heap = DynamicTopK(k=3, key=lambda x: -x)
        heap.add(1)
        heap.add(2)
        heap.add(3)
        heap.add(0)
        assert len(heap.heap) == 3
        assert heap.topk() == [0, 1, 2]

    def test_topk_empty(self):
        heap = DynamicTopK(k=3)
        assert heap.topk() == []

    def test_topk_partial(self):
        heap = DynamicTopK(k=3)
        heap.add(2)
        heap.add(1)
        result = heap.topk()
        assert result == [2, 1]

    def test_topk_full_ordered(self):
        heap = DynamicTopK(k=3)
        values = [1, 3, 2, 5, 4]
        for v in values:
            heap.add(v)
        assert heap.topk() == [5, 4, 3]

    def test_custom_key_complex_objects(self):
        class Item:
            def __init__(self, value):
                self.value = value

        heap = DynamicTopK(k=3, key=lambda x: x.value)
        items = [Item(i) for i in [1, 3, 2, 5, 4]]
        for item in items:
            heap.add(item)

        result = heap.topk()
        assert len(result) == 3
        assert [item.value for item in result] == [5, 4, 3]
