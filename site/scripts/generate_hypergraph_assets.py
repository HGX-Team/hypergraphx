import os
import pathlib
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp())

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from hypergraphx import Hypergraph
from hypergraphx.generation import random_hypergraph
from hypergraphx.measures.degree import degree_sequence
from hypergraphx.representations.projections import clique_projection
from hypergraphx.viz import draw_hypergraph


def normalize(values):
    arr = np.array(values, dtype=float)
    if arr.max() == arr.min():
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())


def save_svg(
    path,
    hypergraph,
    pos,
    node_color,
    node_size,
    edge_color,
    hyperedge_colors,
    edge_width=1.8,
    node_overlay=None,
    draw_kwargs=None,
):
    if draw_kwargs is None:
        draw_kwargs = {}
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    draw_hypergraph(
        hypergraph,
        ax=ax,
        pos=pos,
        edge_color=edge_color,
        hyperedge_color_by_order=hyperedge_colors,
        hyperedge_facecolor_by_order=hyperedge_colors,
        node_size=node_size,
        node_color=node_color,
        node_facecolor="#121417",
        with_node_labels=False,
        hyperedge_alpha=0.35,
        edge_width=edge_width,
        **draw_kwargs,
    )
    if node_overlay:
        nodes = list(hypergraph.get_nodes())
        G = nx.Graph()
        G.add_nodes_from(nodes)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_size=node_overlay["sizes"],
            node_color=node_overlay["colors"],
            edgecolors="#121417",
            ax=ax,
        )
    ax.axis("off")
    output_format = path.suffix.lstrip(".") or "svg"
    fig.savefig(
        path,
        format=output_format,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.2,
        dpi=180,
    )
    plt.close(fig)


def save_simple_hero_svg(path, hyperedge_colors):
    hypergraph = Hypergraph(
        edge_list=[
            (0, 1, 2),
            (2, 3),
            (3, 4, 5),
            (1, 4),
            (0, 2, 4, 6),
            (0, 5),
            (1, 6),
        ]
    )
    draw_hypergraph(hypergraph, edge_color="#121417", node_size=250)
    fig = plt.gcf()
    ax = plt.gca()
    ax.axis("off")
    fig.savefig(
        path, format="svg", transparent=True, bbox_inches="tight", pad_inches=0.2
    )
    plt.close(fig)


def save_degree_histogram_svg(path, degrees: dict, accent: str, ink: str):
    values = list(degrees.values())
    max_degree = max(values) if values else 1
    bins = np.arange(1, max_degree + 2) - 0.5

    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    ax.hist(values, bins=bins, color=accent, alpha=0.85, edgecolor=ink, linewidth=1.2)
    ax.set_xlim(0.5, max_degree + 0.5)

    label_col = ink
    tick_col = ink

    ax.set_title("Degree centrality (counts)", fontsize=11, color=label_col, pad=10)
    ax.set_xlabel("degree", fontsize=9, color=label_col)
    ax.set_ylabel("nodes", fontsize=9, color=label_col)

    ax.tick_params(colors=tick_col)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(axis="y", color=(0, 0, 0, 0.08), linewidth=1)
    ax.set_axisbelow(True)
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    output_format = path.suffix.lstrip(".") or "svg"
    fig.savefig(
        path,
        format=output_format,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.25,
        dpi=180,
    )
    plt.close(fig)


def main():
    seed = 9
    hypergraph = random_hypergraph(
        num_nodes=8,
        num_edges_by_size={2: 4, 3: 4, 4: 4},
        seed=seed,
    )
    pos = nx.spring_layout(clique_projection(hypergraph, keep_isolated=True), seed=seed)

    palette = {
        2: "#e27b4f",
        3: "#24639d",
        4: "#67b79c",
    }

    nodes = list(hypergraph.get_nodes())
    degrees = degree_sequence(hypergraph)
    values = [degrees.get(n, 0) for n in nodes]
    scaled = normalize(values)

    step_dir = pathlib.Path(__file__).resolve().parent.parent / "assets"
    step_dir.mkdir(parents=True, exist_ok=True)

    save_simple_hero_svg(step_dir / "hero-simple-hypergraph.svg", palette)

    hero_hypergraph = Hypergraph(
        edge_list=[(1, 3), (1, 4), (1, 2), (5, 6, 7, 8), (1, 2, 3)]
    )
    save_svg(
        step_dir / "hero-hypergraph.svg",
        hero_hypergraph,
        pos=None,
        node_color="#f2efe9",
        node_size=95,
        edge_color="#3a3a3a",
        hyperedge_colors=palette,
        edge_width=1.2,
        draw_kwargs={"seed": seed},
    )

    save_svg(
        step_dir / "hypergraph-step-1.png",
        hypergraph,
        pos=None,
        node_color="#f2efe9",
        node_size=90,
        edge_color="#3a3a3a",
        hyperedge_colors=palette,
        edge_width=1.1,
        draw_kwargs={"seed": seed},
    )

    save_degree_histogram_svg(
        step_dir / "hypergraph-step-2.png",
        degrees=degrees,
        accent="#e27b4f",
        ink="#121417",
    )

    top2 = set(sorted(degrees, key=degrees.get, reverse=True)[:2])

    save_svg(
        step_dir / "hypergraph-step-3.png",
        hypergraph,
        pos,
        node_color="#f2efe9",
        node_size=70,
        edge_color="#2e2e2e",
        hyperedge_colors=palette,
        node_overlay={
            "sizes": 70 + 80 * scaled,
            "colors": ["#e27b4f" if n in top2 else "#f2efe9" for n in nodes],
        },
    )


if __name__ == "__main__":
    main()
