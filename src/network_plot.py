"""
network_plot.py

Readable single-line diagram of a base-case solution:
nodes grouped by zone, per-node info box (LMP, generation, renewables, load,
net injection) and arrows in the actual flow direction with flow / capacity.

Usage
-----
    from network_plot import plot_network
    bc = solve_base_case(data)
    plot_network(data, bc)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ZONE_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
INK, INK_2 = '#0b0b0b', '#52514e'
BINDING = '#d03b3b'

# hand-placed coordinates for the default 4-node system (zones left -> right)
DEFAULT_POS = {'N2': (0.0, 0.0), 'N1': (5.0, 0.0), 'N3': (13.0, 3.0), 'N4': (13.0, -3.0)}
# side of the node where its info box goes
DEFAULT_BOX = {'N2': 'left', 'N1': 'below', 'N3': 'right', 'N4': 'right'}

_BOX_OFFSET = {'left': (-0.7, 0, 'right', 'center'), 'right': (0.7, 0, 'left', 'center'),
               'above': (0, 0.9, 'center', 'bottom'), 'below': (0, -1.4, 'center', 'top')}


def _layout(data):
    if set(DEFAULT_POS) == set(data.NODES):
        return DEFAULT_POS, DEFAULT_BOX
    import networkx as nx
    pos = {n: tuple(5 * p) for n, p in nx.kamada_kawai_layout(nx.Graph(data.lines)).items()}
    return pos, {n: 'above' if y >= 0 else 'below' for n, (_, y) in pos.items()}


def _node_text(data, bc, n, load, res, price_label='LMP'):
    rows = [f"{price_label:<9} {bc['LMP'][n]:7.2f} €/MWh"]
    for g in data.generators_per_node.get(n, []):
        i = data.gen_idx[g]
        rows.append(f"{g:<4} gen  {bc['g'][g]:5.1f} / {data.gen_max_vec[i]:g} MW"
                    f"  ({data.gen_cost_vec[i]:g} €/MWh)")
    for r in data.renewables_per_node.get(n, []):
        i = data.res_idx[r]
        curt = bc['C'][r]
        rows.append(f"{r:<4} RES  {res[i] - curt:5.1f} / {res[i]:.1f} MW avail."
                    + (f"  curt. {curt:.1f} MW" if curt > 1e-6 else ""))
    d_n = sum(load[data.load_idx[d]] for d in data.loads_per_node.get(n, []))
    rows.append(f"Load      {d_n:7.1f} MW")
    rows.append(f"Net inj.  {bc['P'][n]:+7.1f} MW")
    return "\n".join(rows)


def plot_network(data, bc, load=None, res=None, title=None, ax=None, price_label='LMP'):
    """Draw the network with the base-case results `bc` (output of solve_base_case).

    Any result with the same keys works (g, C, P, f, NP, LMP, cost), e.g. DA or redispatch
    converted with the notebook helper; price_label names the price shown in the node boxes.

    load / res: the load and renewable vectors bc was solved with
                (default: expected DA values, as in solve_base_case).
    """
    load = data.expected_DA_load_vec if load is None else np.asarray(load, float)
    res = data.expected_DA_res_vec if res is None else np.asarray(res, float)
    pos, box_side = _layout(data)

    if ax is None:
        _, ax = plt.subplots(figsize=(15, 8))
    ax.axis('off')

    # zone backgrounds + name
    for k, (z, nodes) in enumerate(data.nodes_per_zone.items()):
        xs, ys = [pos[n][0] for n in nodes], [pos[n][1] for n in nodes]
        c = ZONE_COLORS[k % len(ZONE_COLORS)]
        x0, y0 = min(xs) - 0.8, min(ys) - 0.8
        ax.add_patch(FancyBboxPatch((x0, y0), max(xs) - min(xs) + 1.6, max(ys) - min(ys) + 1.6,
                                    boxstyle='round,pad=0,rounding_size=0.3',
                                    fc=c, ec=c, alpha=0.10, lw=1.5, zorder=0))
        ax.text(x0 + 0.1, y0 - 0.1, f"{z}   NP {bc['NP'][z]:+.1f} MW",
                color=c, fontsize=10.5, fontweight='bold', va='top', zorder=1)

    # lines: arrow in actual flow direction, label beside the arrow
    for k, (a, b) in enumerate(data.lines):
        flow, cap = bc['f'][(a, b)], data.line_cap_vec[k]
        src, dst = (a, b) if flow >= 0 else (b, a)
        loading = abs(flow) / cap
        binding = loading > 0.999
        overload = loading > 1.001
        color = BINDING if binding else INK_2
        ax.add_patch(FancyArrowPatch(pos[src], pos[dst], arrowstyle='-|>', mutation_scale=25,
                                     lw=1.5 + 5 * loading, color=color,
                                     shrinkA=30, shrinkB=30, zorder=2))
        pa, pb = np.array(pos[a]), np.array(pos[b])
        normal = np.array([-(pb - pa)[1], (pb - pa)[0]]) / np.linalg.norm(pb - pa)
        if normal[1] < 0 or (normal[1] == 0 and normal[0] < 0):
            normal = -normal                                   # label above / left of the line
        mx, my = pa + 0.68 * (pb - pa) + 0.55 * normal       # 68 % along, beside the line
        label = f"{a}–{b}:  {abs(flow):.1f} / {cap:g} MW  ({100 * loading:.0f} %)"
        if binding:
            label += "\nOVERLOAD" if overload else "\nBINDING"
        ax.text(mx, my, label, ha='center', va='center', fontsize=10, color=color,
                fontweight='bold' if binding else 'normal', zorder=4,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=color, lw=1))

    # nodes + info boxes
    for n, (x, y) in pos.items():
        c = ZONE_COLORS[data.zone_idx[data.node_zone[n]] % len(ZONE_COLORS)]
        ax.scatter([x], [y], s=1100, c=c, edgecolors='white', linewidths=2, zorder=5)
        ax.text(x, y, n, ha='center', va='center', color='white', fontsize=12,
                fontweight='bold', zorder=6)
        dx, dy, ha, va = _BOX_OFFSET[box_side[n]]
        ax.text(x + dx, y + dy, _node_text(data, bc, n, load, res, price_label), ha=ha, va=va,
                multialignment='left',
                fontsize=9.5, family='monospace', color=INK, zorder=6,
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec=c, lw=1.3))

    ax.set_title(title or f"Base case — total cost {bc['cost']:.1f} €   "
                          f"(arrow = flow direction, width = loading, red = binding)",
                 loc='left', fontsize=12, color=INK)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 6, max(xs) + 7)
    ax.set_ylim(min(ys) - 2, max(ys) + 1.5)
    return ax
