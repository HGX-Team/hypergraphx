import math

from hypergraphx.motifs.utils import generate_motifs


_POS_COLOR = "#4C72B0"
_NEG_COLOR = "#C44E52"
_DEFAULT_BLOB_COLORS = {3: "#9BB8E8", 4: "#F3C7A6", "default": "#A8D5BA"}


def _sort_for_visualization(motifs: list):
    """
    Sort motifs for visualization.
    Motifs are sorted in such a way to show first lower order motifs, then higher order motifs.

    Parameters
    ----------
    motifs : list
        List of motifs to sort

    Returns
    -------
    list
        Sorted list of motifs
    """
    try:
        import numpy as np

        if isinstance(motifs, np.ndarray):
            return np.roll(motifs, 3)
    except Exception:
        pass

    motifs = list(motifs)
    shift = 3 % len(motifs)
    return motifs[-shift:] + motifs[:-shift]


def _default_motif_patterns(order: int = 3):
    mapping, _ = generate_motifs(order)
    return sorted(mapping.keys())


def _style_axes(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, alpha=0.3, linestyle=":", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="both", labelsize=9)


def _node_layout(k: int):
    if k == 1:
        return [(0.5, 0.5)]
    if k == 2:
        return [(0.2, 0.5), (0.8, 0.5)]
    if k == 3:
        h = math.sqrt(3) / 2.0
        return [(0.2, 0.25), (0.8, 0.25), (0.5, 0.25 + h * 0.4)]
    if k == 4:
        return [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    coords = []
    cx, cy, r = 0.5, 0.5, 0.35
    for i in range(k):
        angle = 2 * math.pi * i / k
        coords.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return coords


def _expand_polygon(xs, ys, scale: float = 1.2):
    if not xs or not ys:
        return []
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    pts = []
    for x, y in zip(xs, ys):
        angle = math.atan2(y - cy, x - cx)
        pts.append((angle, x, y))
    pts.sort(key=lambda tup: tup[0])
    vx, vy = [], []
    for angle, x, y in pts:
        dx, dy = x - cx, y - cy
        vx.append(cx + dx * scale)
        vy.append(cy + dy * scale)
    return list(zip(vx, vy))


def _draw_motif_icon(
    ax,
    events,
    center_x: float,
    icon_width: float = 0.85,
    aggregate: bool = False,
    y_center: float = 0.9,
    y_scale: float = 0.3,
    blob_colors: dict | None = None,
    node_color: str = "#222222",
    node_edge: str = "white",
    node_size: float = 30,
    edge_color: str = "#333333",
):
    if not events:
        return
    # Lazy import of Polygon to keep this module import-safe without matplotlib.
    try:
        from matplotlib.patches import Polygon  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plot_motifs requires matplotlib. Install with `pip install hypergraphx[viz]`."
        ) from exc

    node_ids = sorted({n for ev in events for n in ev})
    k = len(node_ids)
    if k == 0:
        return
    base_coords = _node_layout(k)
    jitter = {node: (0.0, 0.0) for node in node_ids}
    icon_width = icon_width * (3 / max(3, k)) ** 0.25
    node_to_xy_local = {node_ids[i]: base_coords[i] for i in range(k)}
    L = len(events)
    if L == 0:
        return
    if blob_colors is None:
        blob_colors = _DEFAULT_BLOB_COLORS

    if aggregate:
        xs_all, ys_all = [], []
        for node in node_ids:
            x_local, y_local = node_to_xy_local[node]
            jx, jy = jitter[node]
            x_scaled = center_x + (x_local + jx - 0.5) * icon_width
            y_scaled = y_center + (y_local + jy - 0.5) * y_scale
            xs_all.append(x_scaled)
            ys_all.append(y_scaled)

        for ev in events:
            ev_set = set(ev)
            xs_present, ys_present = [], []
            for node in node_ids:
                if node in ev_set:
                    x_local, y_local = node_to_xy_local[node]
                    jx, jy = jitter[node]
                    x_scaled = center_x + (x_local + jx - 0.5) * icon_width
                    y_scaled = y_center + (y_local + jy - 0.5) * y_scale
                    xs_present.append(x_scaled)
                    ys_present.append(y_scaled)

            ev_size = len(xs_present)
            if ev_size >= 3:
                poly_pts = _expand_polygon(xs_present, ys_present, scale=1.35)
                if poly_pts:
                    color = blob_colors.get(
                        ev_size, blob_colors.get("default", "#55A868")
                    )
                    ax.add_patch(
                        Polygon(
                            poly_pts,
                            closed=True,
                            facecolor=color,
                            alpha=0.22,
                            edgecolor="#FFFFFF",
                            linewidth=0.4,
                        )
                    )
            elif ev_size == 2:
                x1, x2 = xs_present
                y1, y2 = ys_present
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    linewidth=1.1,
                    color=edge_color,
                    alpha=0.9,
                    solid_capstyle="round",
                )

        if xs_all:
            ax.scatter(
                xs_all,
                ys_all,
                s=node_size * 2.2,
                color=node_edge,
                alpha=0.25,
                linewidths=0,
                zorder=2,
            )
            ax.scatter(
                xs_all,
                ys_all,
                s=node_size,
                color=node_color,
                edgecolors=node_edge,
                linewidths=0.6,
                zorder=4,
            )
    else:
        row_height = 1.0 / L
        for e_idx, ev in enumerate(events):
            ev_set = set(ev)
            row_center_y = 1.0 - (e_idx + 0.5) / L

            xs_all, ys_all = [], []
            xs_present, ys_present = [], []

            for node in node_ids:
                x_local, y_local = node_to_xy_local[node]
                y_scaled = row_center_y + (y_local - 0.5) * (row_height * 0.7)
                x_scaled = center_x + (x_local - 0.5) * icon_width
                xs_all.append(x_scaled)
                ys_all.append(y_scaled)
                if node in ev_set:
                    xs_present.append(x_scaled)
                    ys_present.append(y_scaled)

            ev_size = len(xs_present)
            if ev_size >= 3:
                poly_pts = _expand_polygon(xs_present, ys_present, scale=1.25)
                if poly_pts:
                    if ev_size == 3:
                        color = "#4C72B0"
                    elif ev_size == 4:
                        color = "#DD8452"
                    else:
                        color = "#55A868"
                    ax.add_patch(
                        Polygon(
                            poly_pts,
                            closed=True,
                            facecolor=color,
                            alpha=0.22,
                            edgecolor="none",
                        )
                    )
            elif ev_size == 2:
                x1, x2 = xs_present
                y1, y2 = ys_present
                ax.plot([x1, x2], [y1, y2], linewidth=1.1, color="black", alpha=0.9)

            if xs_all:
                ax.scatter(xs_all, ys_all, s=8, color="0.8", zorder=3)
            if xs_present:
                ax.scatter(xs_present, ys_present, s=12, color="black", zorder=4)


def plot_motifs(
    motifs: list,
    save_name: str = None,
    show: bool = False,
    roman_numbers: bool = False,
    motif_patterns: list = None,
    pos_color: str = _POS_COLOR,
    neg_color: str = _NEG_COLOR,
    blob_colors: dict | None = None,
    icon_width: float = 0.85,
    icon_y: float = 0.86,
    icon_scale: float = 0.95,
    icon_row_ylim: tuple | None = None,
    node_color: str = "#222222",
    node_edge: str = "white",
    node_size: float = 30,
    edge_color: str = "#333333",
    show_motif_labels: bool = False,
    motif_labels: list | None = None,
    annotate_bars: bool = False,
    annotate_fmt: str = "{:+.2f}",
    hover_labels: bool = False,
):
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.patches import Polygon  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plot_motifs requires matplotlib. Install with `pip install hypergraphx[viz]`."
        ) from exc

    """
    Plot motifs. Motifs are sorted in such a way to show first lower order motifs, then higher order motifs.

    Parameters
    ----------
    motifs : list
        List of motif scores or list of (motif, score) pairs.

    save_name : str, optional
        Name of the file to save the plot, by default None
    show : bool
        If True, call plt.show().
    roman_numbers : bool
        If True, use roman numerals on the x-axis instead of motif drawings.
    motif_patterns : list, optional
        List of motif patterns to draw when roman_numbers is False. If None, defaults to
        the canonical order of 3-node motifs.
    pos_color, neg_color : str
        Colors for positive/negative bars.
    blob_colors : dict, optional
        Colors for motif blobs by hyperedge size (keys 3, 4, 'default').
    icon_width, icon_y, icon_scale : float
        Size and placement controls for graphlet icons.
    icon_row_ylim : tuple, optional
        (ymin, ymax) for the icon row. If None, computed from icon_y/scale.
    node_color, node_edge, node_size, edge_color : styling for graphlet nodes/edges.
    show_motif_labels : bool
        If True, show motif labels under graphlets.
    motif_labels : list, optional
        Labels for motifs (length 6). Defaults to M1..M6 when show_motif_labels is True.
    annotate_bars : bool
        If True, annotate bar values above/below bars.
    annotate_fmt : str
        Format string for bar annotations.
    hover_labels : bool
        If True, add interactive hover labels (requires mplcursors).

    Raises
    ------
    ValueError
        Motifs must be a list of length 6.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the bar chart.
    """
    if len(motifs) == 0:
        raise ValueError("Motifs must be a non-empty list.")
    if all(isinstance(m, (list, tuple)) and len(m) >= 2 for m in motifs):
        motifs = [m[1] for m in motifs]
    motifs = [float(x) for x in motifs]
    if len(motifs) != 6:
        raise ValueError("Motifs must be a list of length 6.")
    if motif_labels is not None and len(motif_labels) != len(motifs):
        raise ValueError("motif_labels length must match motifs length.")
    motifs = _sort_for_visualization(motifs)
    if motif_patterns is None:
        motif_patterns = _default_motif_patterns(order=3)
    if len(motif_patterns) != len(motifs):
        raise ValueError("motif_patterns length must match motifs length.")
    motif_patterns = _sort_for_visualization(motif_patterns)
    cols = [neg_color if (x < 0) else pos_color for x in motifs]
    labels = ["I", "II", "III", "IV", "V", "VI"]
    if roman_numbers:
        fig = plt.gcf()
        ax_bar = fig.add_subplot(111)
        idx = list(range(len(motifs)))
        _style_axes(ax_bar, grid_axis=None)
        bars = ax_bar.bar(
            idx,
            motifs,
            color=cols,
            width=0.85,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax_bar.axhline(0, color="black", linewidth=0.5)
        ax_bar.set_ylabel("Motif abundance score")
        ax_bar.set_xticks(idx)
        ax_bar.set_xticklabels(labels)
        ax_bar.set_ylim(-1, 1)
        ax_bar.set_xlim(-0.5, len(motifs) - 0.5)
        if annotate_bars:
            for i, val in enumerate(motifs):
                ax_bar.text(
                    i,
                    val + (0.03 if val >= 0 else -0.03),
                    annotate_fmt.format(val),
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=8,
                    color="#333333",
                )
        if hover_labels:
            try:
                import mplcursors

                labels = motif_labels or [f"M{i+1}" for i in range(len(motifs))]
                cursor = mplcursors.cursor(bars, hover=True)

                @cursor.connect("add")
                def _on_add(sel):
                    idx = sel.index
                    sel.annotation.set_text(f"{labels[idx]}: {motifs[idx]:.3f}")

            except Exception:
                pass
    else:
        fig = plt.figure(figsize=(max(8, 0.6 * len(motifs)), 5))
        gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 0.6], hspace=0.01)

        ax_bar = fig.add_subplot(gs[0])
        idx = list(range(len(motifs)))
        _style_axes(ax_bar, grid_axis=None)
        bars = ax_bar.bar(
            idx,
            motifs,
            color=cols,
            width=0.85,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax_bar.axhline(0, color="black", linewidth=0.5)
        ax_bar.set_ylim(-1, 1)
        ax_bar.set_ylabel("Motif abundance score")
        ax_bar.set_xticks(idx)
        ax_bar.set_xticklabels([""] * len(idx))
        ax_bar.tick_params(axis="x", bottom=True, length=4, width=0.8, color="#333333")
        ax_bar.set_xlim(-0.5, len(motifs) - 0.5)
        if annotate_bars:
            for i, val in enumerate(motifs):
                ax_bar.text(
                    i,
                    val + (0.03 if val >= 0 else -0.03),
                    annotate_fmt.format(val),
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=8,
                    color="#333333",
                )
        if hover_labels:
            try:
                import mplcursors

                labels = motif_labels or [f"M{i+1}" for i in range(len(motifs))]
                cursor = mplcursors.cursor(bars, hover=True)

                @cursor.connect("add")
                def _on_add(sel):
                    idx = sel.index
                    sel.annotation.set_text(f"{labels[idx]}: {motifs[idx]:.3f}")

            except Exception:
                pass

        ax_icon = fig.add_subplot(gs[1])
        if icon_row_ylim is None:
            icon_row_ylim = (
                icon_y - 0.5 * icon_scale - 0.1,
                icon_y + 0.5 * icon_scale + 0.1,
            )
        ax_icon.set_ylim(*icon_row_ylim)
        ax_icon.set_xlim(-0.5, len(motif_patterns) - 0.5)
        ax_icon.set_yticks([])
        ax_icon.set_xticks([])
        ax_icon.tick_params(axis="x", bottom=False, top=False, labelbottom=False)
        for spine in ax_icon.spines.values():
            spine.set_visible(False)
        ax_icon.patch.set_alpha(0)
        ax_icon.set_zorder(0)
        ax_bar.set_zorder(2)

        for i, pattern in enumerate(motif_patterns):
            _draw_motif_icon(
                ax_icon,
                pattern,
                center_x=i,
                icon_width=icon_width,
                aggregate=True,
                y_center=icon_y,
                y_scale=icon_scale,
                blob_colors=blob_colors,
                node_color=node_color,
                node_edge=node_edge,
                node_size=node_size,
                edge_color=edge_color,
            )
        if show_motif_labels:
            labels = motif_labels or [f"M{i+1}" for i in range(len(motifs))]
            label_y = icon_row_ylim[0] + 0.02
            for i, label in enumerate(labels):
                ax_icon.text(
                    i,
                    label_y,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#333333",
                )
        fig.subplots_adjust(hspace=0.01, bottom=0.08, top=0.97)
    if save_name != None:
        plt.savefig("{}".format(save_name), bbox_inches="tight")
    if show:
        plt.show()
    return ax_bar
