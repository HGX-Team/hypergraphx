🚀 Quickstart
========================================

Common tasks and minimal examples. Copy and run.

Tasks
-----

- Create and query: :doc:`api/hypergraphx.core`
- Load and save: :doc:`api/hypergraphx.readwrite`
- Measures: :doc:`api/hypergraphx.measures`
- Projections: :doc:`api/hypergraphx.representations`
- Matrices: :doc:`api/hypergraphx.linalg`

Core classes
------------

- ``hypergraphx.Hypergraph`` (undirected)
- ``hypergraphx.DirectedHypergraph``
- ``hypergraphx.TemporalHypergraph``
- ``hypergraphx.MultiplexHypergraph``

Create a hypergraph
-------------------

.. code-block:: python

   import hypergraphx as hgx

   hg = hgx.Hypergraph(edge_list=[(0, 1, 2), (2, 3)], weighted=True, weights=[1.0, 2.0])
   hg.add_edge((3, 4), weight=1.5, metadata={"kind": "tri"})
   print(hg)

.. note::

   Hypergraphs are weighted by default. If you want an unweighted hypergraph, pass
   ``weighted=False`` (unweighted graphs only accept ``weight=None`` or ``weight=1``).

Directed / temporal / multiplex
-------------------------------

.. code-block:: python

   dh = hgx.DirectedHypergraph(edge_list=[((0, 1), (2,)), ((2,), (3,))])
   th = hgx.TemporalHypergraph(edge_list=[(0, (0, 1)), (1, (1, 2))])
   mx = hgx.MultiplexHypergraph(edge_list=[(0, 1), (1, 2)], edge_layer=["L1", "L2"])

   # Directed in/out degrees
   dh.in_degree(2)
   dh.out_degree(2)

Common queries
--------------

.. code-block:: python

   hg.num_nodes()
   hg.num_edges()
   hg.get_edges()
   hg.get_incident_edges(2)
   hg.degree(2)
   hg.get_edge_metadata((0, 1, 2))

.. tip::

   For a smaller API surface, start with ``hypergraphx.core`` and ``hypergraphx.readwrite``.

Common patterns
---------------

.. code-block:: python

   # Build -> analyze -> save
   hg = hgx.Hypergraph(edge_list=[(0, 1), (1, 2, 3)])
   seq = degree_sequence(hg)
   hgx.save_hypergraph(hg, "graph.json")

Conversions
-----------

.. code-block:: python

   # Drop direction, time, or layer information
   dh = hgx.DirectedHypergraph(edge_list=[((0, 1), (2,))], weighted=True, weights=[2.0])
   th = hgx.TemporalHypergraph(edge_list=[(0, (0, 1)), (1, (0, 1))], weighted=True, weights=[1.0, 2.0])
   mx = hgx.MultiplexHypergraph(edge_list=[(0, 1), (0, 1)], edge_layer=["L1", "L2"], weighted=True, weights=[1.0, 2.0])

   dh.to_hypergraph()  # merges source/target, sums weights
   th.to_hypergraph()  # drops time, sums weights
   mx.to_hypergraph()  # drops layer, sums weights

Load and save
-------------

.. code-block:: python

   hgx.save_hypergraph(hg, "graph.json")
   hg2 = hgx.load_hypergraph("graph.json")

   # Server datasets
   # Network loading is opt-in (so offline / sandboxed environments don't surprise you)
   hg3 = hgx.load_hypergraph_from_server("toy", fmt="json", allow_network=True)

Projections and matrices
------------------------

.. code-block:: python

   from hypergraphx.linalg import adjacency_matrix
   from hypergraphx.representations.projections import clique_projection

   A = adjacency_matrix(hg)
   G = clique_projection(hg)

Measures
--------

.. code-block:: python

   from hypergraphx.measures.degree import degree_sequence
   from hypergraphx.measures.shortest_paths import calc_ho_shortest_paths
   seq = degree_sequence(hg)
   paths = calc_ho_shortest_paths(hg)

API map
-------

- Core classes: ``hypergraphx.core``
- Read/write: ``hypergraphx.readwrite``
- Measures: ``hypergraphx.measures``
- Representations: ``hypergraphx.representations``
- Linear algebra: ``hypergraphx.linalg``
- Filters: ``hypergraphx.filters``
- Motifs: ``hypergraphx.motifs``
- Dynamics: ``hypergraphx.dynamics``
- Communities: ``hypergraphx.communities``
- Utils: ``hypergraphx.utils``

Exceptions
----------

All custom exceptions inherit from ``HypergraphxError``. Most also inherit from
``ValueError`` (``ReadwriteError`` inherits from ``RuntimeError``).

.. code-block:: python

   from hypergraphx import MissingNodeError

   try:
       hg.get_incident_edges("missing")
   except MissingNodeError:
       pass

Logging (optional)
------------------

Hypergraphx uses the standard logging module. To see info-level logs:

.. code-block:: python

   import logging

   logging.basicConfig(level=logging.INFO)
