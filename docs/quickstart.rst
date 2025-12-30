Quickstart
==========

Common tasks and minimal examples. Copy and run.

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

Directed / temporal / multiplex
-------------------------------

.. code-block:: python

   dh = hgx.DirectedHypergraph(edge_list=[((0, 1), (2,)), ((2,), (3,))])
   th = hgx.TemporalHypergraph(edge_list=[(0, (0, 1)), (1, (1, 2))])
   mx = hgx.MultiplexHypergraph(edge_list=[(0, 1), (1, 2)], edge_layer=["L1", "L2"])

Common queries
--------------

.. code-block:: python

   hg.num_nodes()
   hg.num_edges()
   hg.get_edges()
   hg.get_incident_edges(2)
   hg.degree(2)
   hg.get_edge_metadata((0, 1, 2))

Load and save
-------------

.. code-block:: python

   hgx.save_hypergraph(hg, "graph.json")
   hg2 = hgx.load_hypergraph("graph.json")

   # Server datasets
   hg3 = hgx.load_hypergraph_from_server("toy", fmt="json")

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

All custom exceptions inherit from ``HypergraphxError`` and ``ValueError``.

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
