"""
power_system_data.py

Data container for a zonal / nodal power system, intended as the input layer
for an (object oriented) optimisation model.

Usage
-----
    from power_system_data import PowerSystemData

    data = PowerSystemData()                      # defaults = the 4-node, 3-zone system
    data = PowerSystemData(inter_zone_capacity=8) # override any input field

    access through data.NODES, data.gen_max_capacity, data.PTDF, data.GSK, ...

Every input field defaults to None / a value; anything left as None is filled
in with the default system in __post_init__, and all derived data (mappings,
matrices, GSK, PTDF) is computed from the final inputs.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# Helper functions (pure, independent of the data class)
def _to_dict(M: np.ndarray, rows: list, cols: list) -> dict:
    """2-D array -> nested dict {row_label: {col_label: value}}."""
    return {r: {c: M[i, j].item() for j, c in enumerate(cols)}
            for i, r in enumerate(rows)}


def build_connection_matrices(NODES, GENERATORS, LOADS, ZONES,
                              connections_per_node, generators_per_node,
                              loads_per_node, nodes_per_zone) -> dict:
    """Adjacency / mapping / incidence matrices (NumPy arrays + line lists)."""
    n_idx = {n: i for i, n in enumerate(NODES)}
    g_idx = {g: i for i, g in enumerate(GENERATORS)}
    d_idx = {d: i for i, d in enumerate(LOADS)}
    z_idx = {z: i for i, z in enumerate(ZONES)}

    # node x node adjacency
    A_nn = np.zeros((len(NODES), len(NODES)), dtype=int)
    for n, nbrs in connections_per_node.items():
        for m in nbrs:
            A_nn[n_idx[n], n_idx[m]] = 1
    assert (A_nn == A_nn.T).all(), "connections_per_node is not symmetric"

    # node x generator, node x load
    M_ng = np.zeros((len(NODES), len(GENERATORS)), dtype=int)
    for n, gens in generators_per_node.items():
        for g in gens:
            M_ng[n_idx[n], g_idx[g]] = 1

    M_nd = np.zeros((len(NODES), len(LOADS)), dtype=int)
    for n, loads in loads_per_node.items():
        for d in loads:
            M_nd[n_idx[n], d_idx[d]] = 1

    # zone x node
    M_zn = np.zeros((len(ZONES), len(NODES)), dtype=int)
    for z, nodes in nodes_per_zone.items():
        for n in nodes:
            M_zn[z_idx[z], n_idx[n]] = 1

    # zone-level mappings obtained by chaining
    M_zg = (M_zn @ M_ng > 0).astype(int)
    M_zd = (M_zn @ M_nd > 0).astype(int)
    A_zz = (M_zn @ A_nn @ M_zn.T > 0).astype(int)
    np.fill_diagonal(A_zz, 0)

    # lines: each undirected line once, ordered by node index
    node_zone = {n: z for z, ns in nodes_per_zone.items() for n in ns}
    lines = [(NODES[i], NODES[j])
             for i in range(len(NODES)) for j in range(i + 1, len(NODES))
             if A_nn[i, j] == 1]
    inter_lines = [l for l in lines if node_zone[l[0]] != node_zone[l[1]]]
    intra_lines = [l for l in lines if node_zone[l[0]] == node_zone[l[1]]]

    # node x line incidence (+1 at "from" node, -1 at "to" node)
    B_nl = np.zeros((len(NODES), len(lines)), dtype=int)
    for k, (n, m) in enumerate(lines):
        B_nl[n_idx[n], k] = 1
        B_nl[n_idx[m], k] = -1

    # zone x inter-zone-line incidence
    B_zl = np.zeros((len(ZONES), len(inter_lines)), dtype=int)
    for k, (n, m) in enumerate(inter_lines):
        B_zl[z_idx[node_zone[n]], k] = 1
        B_zl[z_idx[node_zone[m]], k] = -1

    return dict(A_nn=A_nn, M_ng=M_ng, M_nd=M_nd, M_zn=M_zn, M_zg=M_zg,
                M_zd=M_zd, A_zz=A_zz, B_nl=B_nl, B_zl=B_zl,
                lines=lines, intra_lines=intra_lines, inter_lines=inter_lines,
                node_zone=node_zone)


def build_gsk(NODES, ZONES, nodes_per_zone, generators_per_node=None,
              gen_max_capacity=None, loads_per_node=None,
              expected_DA_load=None, method: str = 'generation') -> np.ndarray:
    """Generation shift key GSK[node, zone]; every column sums to 1."""
    n_idx = {n: i for i, n in enumerate(NODES)}
    GSK = np.zeros((len(NODES), len(ZONES)))

    if method == 'flat':
        w = {n: 1.0 for n in NODES}
    elif method == 'generation':
        w = {n: sum(gen_max_capacity[g] for g in generators_per_node[n])
             for n in NODES}
    elif method == 'load':
        w = {n: sum(expected_DA_load[d] for d in loads_per_node[n])
             for n in NODES}
    else:
        raise ValueError(f"unknown GSK method '{method}'")

    for z_i, z in enumerate(ZONES):
        nodes = nodes_per_zone[z]
        total = sum(w[n] for n in nodes)
        for n in nodes:
            GSK[n_idx[n], z_i] = w[n] / total if total > 0 else 1.0 / len(nodes)

    assert np.allclose(GSK.sum(axis=0), 1.0), "each GSK column must sum to 1"
    return GSK


def build_nodal_ptdf(NODES, B_nl, line_susceptance_vec, slack_bus):
    """
    DC PTDF (lines x nodes) and bus susceptance matrix (nodes x nodes).
    Positive flow = from first to second node of the line tuple.
    Injections: positive = generation, negative = load.
    """
    b = np.asarray(line_susceptance_vec, dtype=float)
    B_bus = B_nl @ np.diag(b) @ B_nl.T

    s = NODES.index(slack_bus)
    keep = [i for i in range(len(NODES)) if i != s]
    X_red = np.linalg.inv(B_bus[np.ix_(keep, keep)])

    PTDF = np.zeros((B_nl.shape[1], len(NODES)))
    PTDF[:, keep] = np.diag(b) @ B_nl.T[:, keep] @ X_red
    return PTDF, B_bus


# Main data class
@dataclass
class PowerSystemData:
    # sizes
    N_GENERATORS: int = 5
    N_LOADS: int = 4
    N_NODES: int = 4
    N_ZONES: int = 3
    N_SCENARIOS: int = 2

    # sets 
    GENERATORS: list | None = None
    LOADS: list | None = None
    NODES: list | None = None
    ZONES: list | None = None
    SCENARIOS: list | None = None

    # general 
    S_BASE_MVA: float = 100.0
    V_BASE_KV: dict | None = None
    time_step: int = 1
    slack_bus: str | None = None

    # topology
    nodes_per_zone: dict | None = None
    connections_per_node: dict | None = None
    generators_per_node: dict | None = None
    loads_per_node: dict | None = None

    # generators 
    gen_linear_cost_coeff: dict | None = None
    gen_down_deviation_cost_coeff: dict | None = None
    gen_up_deviation_cost_coeff: dict | None = None
    gen_max_capacity: dict | None = None
    gen_min_capacity: dict | None = None

    # loads / scenarios 
    load_max_capacity: dict | None = None
    load_min_capacity: dict | None = None
    load_scenarios: dict | None = None
    scenario_probability: dict | None = None
    expected_DA_load: dict | None = None

    # network parameters 
    intra_zone_capacity: float = 10.0          # line N1-N2
    inter_zone_capacity: float = 20.0          # rest of lines
    intra_zone_susceptance: float = 2.5
    inter_zone_susceptance: float = 2.5
    line_capacity_per_line: dict | None = None     # {(n, m): cap}   override
    line_susceptance_per_line: dict | None = None  # {(n, m): b}     override

    # GSK 
    gsk_method: str = 'generation'            # 'flat' | 'generation' | 'load'

    def __post_init__(self):
        self._set_defaults()
        self._build_mappings()
        self._build_network()
        self._build_gsk_and_ptdf()
        self._build_vectors()
        self.validate()

    def _default(self, name, factory):
        """Fill attribute `name` with factory() if it is still None."""
        if getattr(self, name) is None:
            setattr(self, name, factory())

    def _set_defaults(self):
        # sets
        self._default('GENERATORS', lambda: [f'G{i+1}' for i in range(self.N_GENERATORS)])
        self._default('LOADS',      lambda: [f'D{i+1}' for i in range(self.N_LOADS)])
        self._default('NODES',      lambda: [f'N{i+1}' for i in range(self.N_NODES)])
        self._default('ZONES',      lambda: [f'Z{i+1}' for i in range(self.N_ZONES)])
        self._default('SCENARIOS',  lambda: [f'S{i+1}' for i in range(self.N_SCENARIOS)])
        self.indexes = dict(GENERATORS=self.GENERATORS, 
                            LOADS=self.LOADS,
                            NODES=self.NODES, 
                            ZONES=self.ZONES,
                            SCENARIOS=self.SCENARIOS)

        self._default('V_BASE_KV', lambda: {n: 230.0 for n in self.NODES})
        self._default('slack_bus', lambda: self.NODES[0])

        # topology: Zone 1 = {N1, N2}, Zone 2 = {N3}, Zone 3 = {N4}
        self._default('nodes_per_zone',
                      lambda: {'Z1': ['N1', 'N2'], 'Z2': ['N3'], 'Z3': ['N4']})
        self._default('connections_per_node', lambda: {
            'N1': ['N2', 'N3', 'N4'],
            'N2': ['N1'],
            'N3': ['N1', 'N4'],
            'N4': ['N1', 'N3'],
        })
        self._default('generators_per_node', lambda: {
            'N1': ['G1', 'G2'], 'N2': ['G3'], 'N3': ['G4'], 'N4': ['G5'],
        })
        self._default('loads_per_node', lambda: {
            'N1': ['D1'], 'N2': ['D2'], 'N3': ['D3'], 'N4': ['D4'],
        })

        # generators: "cap @ cost ± deviation cost"
        self._default('gen_linear_cost_coeff',
                      lambda: {'G1': 9, 'G2': 20, 'G3': 10, 'G4': 11, 'G5': 12})
        self._default('gen_down_deviation_cost_coeff',
                      lambda: {'G1': 2, 'G2': 0, 'G3': 2, 'G4': 2, 'G5': 2})
        self._default('gen_up_deviation_cost_coeff',
                      lambda: {'G1': 2, 'G2': 0, 'G3': 2, 'G4': 2, 'G5': 2})
        self._default('gen_max_capacity',
                      lambda: {'G1': 20, 'G2': 20, 'G3': 10, 'G4': 25, 'G5': 10})
        self._default('gen_min_capacity', lambda: {g: 0 for g in self.GENERATORS})

        # loads 
        self._default('load_max_capacity',
                      lambda: {'D1': 150, 'D2': 100, 'D3': 100, 'D4': 100})
        self._default('load_min_capacity', lambda: {d: 0 for d in self.LOADS})
        self._default('load_scenarios', lambda: {
            'D1': {'S1': 2, 'S2': 7},
            'D2': {'S1': 6, 'S2': 1},
            'D3': {'S1': 3, 'S2': 4},
            'D4': {'S1': 4, 'S2': 2},
        })
        self._default('scenario_probability', lambda: {'S1': 0.6, 'S2': 0.4})
        self._default('expected_DA_load', lambda: {
            l: sum(self.scenario_probability[s] * self.load_scenarios[l][s]
                   for s in self.SCENARIOS)
            for l in self.LOADS
        })

    def _build_mappings(self):
        m = build_connection_matrices(
            self.NODES, self.GENERATORS, self.LOADS, self.ZONES,
            self.connections_per_node, self.generators_per_node,
            self.loads_per_node, self.nodes_per_zone)
        self.mats = m

        # index lookups (handy for building numpy / solver variables)
        self.node_idx = {n: i for i, n in enumerate(self.NODES)}
        self.gen_idx = {g: i for i, g in enumerate(self.GENERATORS)}
        self.load_idx = {d: i for i, d in enumerate(self.LOADS)}
        self.zone_idx = {z: i for i, z in enumerate(self.ZONES)}
        self.scenario_idx = {s: i for i, s in enumerate(self.SCENARIOS)}

        self.node_zone = m['node_zone']                      # node -> zone

        # arrays
        self.A_nn, self.M_ng, self.M_nd = m['A_nn'], m['M_ng'], m['M_nd']
        self.M_zn, self.M_zg, self.M_zd = m['M_zn'], m['M_zg'], m['M_zd']
        self.A_zz, self.B_nl, self.B_zl = m['A_zz'], m['B_nl'], m['B_zl']

        # nested-dict versions
        N, G, D, Z = self.NODES, self.GENERATORS, self.LOADS, self.ZONES
        self.lines = m['lines']
        self.intra_lines = m['intra_lines']
        self.inter_lines = m['inter_lines']
        self.mapping_buses = _to_dict(self.A_nn, N, N)
        self.mapping_generators = _to_dict(self.M_ng, N, G)
        self.mapping_loads = _to_dict(self.M_nd, N, D)
        self.mapping_zones = _to_dict(self.M_zn, Z, N)
        self.mapping_zone_generators = _to_dict(self.M_zg, Z, G)
        self.mapping_zone_loads = _to_dict(self.M_zd, Z, D)
        self.mapping_zone_zone = _to_dict(self.A_zz, Z, Z)

        # connection lists (kept for backwards compatibility)
        self.all_connections = [(n, k) for n in N for k in self.connections_per_node[n]]
        self.processed_connections = list(self.lines)
        self.intra_zone_connections = list(self.intra_lines)
        self.inter_zone_connections = list(self.inter_lines)

        # reverse lookups
        self.zone_nodes = self.nodes_per_zone
        self.zone_generators = {z: [g for n in ns for g in self.generators_per_node[n]]
                                for z, ns in self.nodes_per_zone.items()}
        self.zone_loads = {z: [d for n in ns for d in self.loads_per_node[n]]
                           for z, ns in self.nodes_per_zone.items()}
        self.node_of_generator = {g: n for n, gs in self.generators_per_node.items() for g in gs}
        self.node_of_load = {d: n for n, ds in self.loads_per_node.items() for d in ds}

    def _build_network(self):
        # per-line capacity and susceptance (ordered like self.lines)
        def is_inter(l):
            return self.node_zone[l[0]] != self.node_zone[l[1]]

        cap = {l: (self.inter_zone_capacity if is_inter(l) else self.intra_zone_capacity)
               for l in self.lines}
        sus = {l: (self.inter_zone_susceptance if is_inter(l) else self.intra_zone_susceptance)
               for l in self.lines}
        if self.line_capacity_per_line:
            cap.update(self.line_capacity_per_line)
        if self.line_susceptance_per_line:
            sus.update(self.line_susceptance_per_line)
        self.line_capacity_per_line = cap
        self.line_susceptance_per_line = sus

        self.line_cap_vec = np.array([cap[l] for l in self.lines], dtype=float)
        self.line_b_vec = np.array([sus[l] for l in self.lines], dtype=float)

        # nodal-matrix versions
        N = self.NODES
        self.line_capacity = {n: {m: 0 for m in N} for n in N}
        for (n, m), c in cap.items():
            self.line_capacity[n][m] = c
            self.line_capacity[m][n] = c

    def _build_gsk_and_ptdf(self):
        self.GSK = build_gsk(self.NODES, self.ZONES, self.nodes_per_zone,
                             self.generators_per_node, self.gen_max_capacity,
                             self.loads_per_node, self.expected_DA_load,
                             method=self.gsk_method)
        self.gsk = _to_dict(self.GSK, self.NODES, self.ZONES)

        self.PTDF, self.B_bus = build_nodal_ptdf(
            self.NODES, self.B_nl, self.line_b_vec, self.slack_bus)
        self.PTDF_zonal = self.PTDF @ self.GSK

        self.bus_susceptance = _to_dict(self.B_bus, self.NODES, self.NODES)
        self.ptdf = _to_dict(self.PTDF, self.lines, self.NODES)
        self.ptdf_zonal = _to_dict(self.PTDF_zonal, self.lines, self.ZONES)

    def _build_vectors(self):
        G, D, S = self.GENERATORS, self.LOADS, self.SCENARIOS
        self.gen_cost_vec = np.array([self.gen_linear_cost_coeff[g] for g in G], float)
        self.gen_up_cost_vec = np.array([self.gen_up_deviation_cost_coeff[g] for g in G], float)
        self.gen_down_cost_vec = np.array([self.gen_down_deviation_cost_coeff[g] for g in G], float)
        self.gen_max_vec = np.array([self.gen_max_capacity[g] for g in G], float)
        self.gen_min_vec = np.array([self.gen_min_capacity[g] for g in G], float)
        self.load_max_vec = np.array([self.load_max_capacity[d] for d in D], float)
        self.load_min_vec = np.array([self.load_min_capacity[d] for d in D], float)
        self.expected_DA_load_vec = np.array([self.expected_DA_load[d] for d in D], float)
        self.scenario_prob_vec = np.array([self.scenario_probability[s] for s in S], float)
        self.load_scenarios_mat = np.array(
            [[self.load_scenarios[d][s] for s in S] for d in D], float)

    def validate(self):
        assert len(self.GENERATORS) == len(self.gen_max_capacity), "generator data / set size mismatch"
        assert set(self.node_zone) == set(self.NODES), "every node must belong to exactly one zone"
        assert sum(len(v) for v in self.nodes_per_zone.values()) == len(self.NODES), \
            "a node appears in more than one zone"
        assert self.slack_bus in self.NODES, "slack_bus must be a node"
        assert abs(sum(self.scenario_probability.values()) - 1.0) < 1e-9, \
            "scenario probabilities must sum to 1"
        for g in self.GENERATORS:
            assert g in self.node_of_generator, f"generator {g} is not placed at any node"
        for d in self.LOADS:
            assert d in self.node_of_load, f"load {d} is not placed at any node"
        # B_bus from the incidence matrix must equal the Laplacian built from line data
        assert np.allclose(self.B_bus.sum(axis=1), 0), "B_bus rows must sum to zero"
        assert np.allclose(self.GSK.sum(axis=0), 1.0), "GSK columns must sum to 1"

    # -------------------------------------------------------------------------
    def summary(self):
        print(f"Nodes      : {self.NODES}  (slack: {self.slack_bus})")
        print(f"Zones      : {self.nodes_per_zone}")
        print(f"Generators : {self.generators_per_node}")
        print(f"Loads      : {self.loads_per_node}")
        print(f"Lines      : {self.lines}")
        print(f"  intra    : {self.intra_lines}")
        print(f"  inter    : {self.inter_lines}")
        print(f"Capacities : {dict(zip(self.lines, self.line_cap_vec))}")
        print(f"GSK method : {self.gsk_method}")
        np.set_printoptions(precision=3, suppress=True)
        print("B (nodes x nodes):\n", self.B_bus)
        print("GSK (nodes x zones):\n", self.GSK)
        print("PTDF (lines x nodes):\n", self.PTDF)
        print("Zonal PTDF (lines x zones):\n", self.PTDF_zonal)


# =============================================================================
if __name__ == "__main__":
    data = PowerSystemData(slack_bus='N4')
    data.summary()
