from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Optional, Tuple


class NodeView:
    """
    Lightweight, dynamic view over a hypergraph's nodes.

    Behaves like a container:
    - iteration yields node labels
    - len(view) returns current number of nodes
    - `node in view` checks membership
    """

    def __init__(self, hypergraph: Any):
        self._h = hypergraph

    def __iter__(self) -> Iterator[Any]:
        yield from self._h.get_nodes()

    def __len__(self) -> int:
        return len(self._h.get_nodes())

    def __contains__(self, node: Any) -> bool:
        return self._h.check_node(node)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n={len(self)})"


@dataclass(frozen=True)
class _EdgeViewFilters:
    order: Optional[int] = None
    up_to: bool = False
    time_window: Optional[Tuple[int, int]] = None
    layer: Optional[str] = None


class EdgeView:
    """
    Lightweight, dynamic view over a hypergraph's edges (edge keys).

    Filtering returns new views without copying the underlying hypergraph.
    """

    def __init__(self, hypergraph: Any, filters: Optional[_EdgeViewFilters] = None):
        self._h = hypergraph
        self._f = filters or _EdgeViewFilters()

    def _iter_edge_keys(self) -> Iterator[Any]:
        edges: Iterable[Any] = self._h._edge_list.keys()

        if self._f.time_window is not None:
            t0, t1 = self._f.time_window
            for edge_key in edges:
                if (
                    isinstance(edge_key, tuple)
                    and len(edge_key) == 2
                    and isinstance(edge_key[0], int)
                    and t0 <= edge_key[0] < t1
                ):
                    yield edge_key
            return

        if self._f.layer is not None:
            layer = self._f.layer
            for edge_key in edges:
                if (
                    isinstance(edge_key, tuple)
                    and len(edge_key) == 2
                    and edge_key[0] == layer
                ):
                    yield edge_key
            return

        yield from edges

    def __iter__(self) -> Iterator[Any]:
        edges = list(self._iter_edge_keys())
        if self._f.order is None:
            yield from edges
        else:
            yield from self._h._filter_edges_by_order(
                edges, order=self._f.order, up_to=self._f.up_to
            )

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __contains__(self, edge: Any) -> bool:
        try:
            edge_key = self._h._normalize_edge(edge)
        except Exception:
            edge_key = edge

        if not self._h._edge_exists(edge_key):
            return False

        if self._f.time_window is not None:
            if not (
                isinstance(edge_key, tuple)
                and len(edge_key) == 2
                and isinstance(edge_key[0], int)
            ):
                return False
            t0, t1 = self._f.time_window
            if not (t0 <= edge_key[0] < t1):
                return False

        if self._f.layer is not None:
            if not (isinstance(edge_key, tuple) and len(edge_key) == 2):
                return False
            if edge_key[0] != self._f.layer:
                return False

        if self._f.order is not None:
            if self._f.up_to:
                if self._h._edge_order(edge_key) > self._f.order:
                    return False
            else:
                if self._h._edge_order(edge_key) != self._f.order:
                    return False

        return True

    def order(self, order: int, up_to: bool = False) -> EdgeView:
        if self._f.order is not None:
            raise ValueError("EdgeView already has an order/size filter applied.")
        return EdgeView(
            self._h,
            _EdgeViewFilters(
                order=order,
                up_to=up_to,
                time_window=self._f.time_window,
                layer=self._f.layer,
            ),
        )

    def size(self, size: int, up_to: bool = False) -> EdgeView:
        return self.order(size - 1, up_to=up_to)

    def time_window(self, window: Tuple[int, int]) -> EdgeView:
        from hypergraphx.core.temporal import TemporalHypergraph

        if not isinstance(self._h, TemporalHypergraph):
            raise TypeError("time_window() is only available for TemporalHypergraph.")
        if self._f.time_window is not None:
            raise ValueError("EdgeView already has a time_window filter applied.")
        if not (isinstance(window, tuple) and len(window) == 2):
            raise TypeError("time_window must be a (start, end) tuple.")
        return EdgeView(
            self._h,
            _EdgeViewFilters(
                order=self._f.order,
                up_to=self._f.up_to,
                time_window=window,
                layer=self._f.layer,
            ),
        )

    def layer(self, layer: str) -> EdgeView:
        from hypergraphx.core.multiplex import MultiplexHypergraph

        if not isinstance(self._h, MultiplexHypergraph):
            raise TypeError("layer() is only available for MultiplexHypergraph.")
        if self._f.layer is not None:
            raise ValueError("EdgeView already has a layer filter applied.")
        return EdgeView(
            self._h,
            _EdgeViewFilters(
                order=self._f.order,
                up_to=self._f.up_to,
                time_window=self._f.time_window,
                layer=layer,
            ),
        )

    def __repr__(self) -> str:
        parts = []
        if self._f.order is not None:
            parts.append(f"order={'<=' if self._f.up_to else '=='}{self._f.order}")
        if self._f.time_window is not None:
            parts.append(f"time_window={self._f.time_window}")
        if self._f.layer is not None:
            parts.append(f"layer={self._f.layer!r}")
        joined = ", ".join(parts) if parts else "all"
        return f"{self.__class__.__name__}({joined})"
