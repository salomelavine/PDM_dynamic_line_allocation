"""
power_system_data.py

Data container for a zonal / nodal power system, intended as the input layer
for an (object oriented) optimisation model.

Usage
-----
    from power_system_data import PowerSystemData

    data = PowerSystemData()                      # defaults = the 4-node, 3-zone system
    data = PowerSystemData(inter_zone_capacity=8) # override any input field
    data = PowerSystemData.from_csv('../data/base_case')            # read everything from CSV files
    data = PowerSystemData.from_csv('../data/base_case', time='T1') # pick one time step of the profiles
    data.to_csv('../data/my_case')                # write the current data as CSV templates

    access through data.NODES, data.gen_max_capacity, data.PTDF, data.GSK, ...

Every input field defaults to None / a value; anything left as None is filled
in with the default system in __post_init__, and all derived data (mappings,
matrices, GSK, PTDF) is computed from the final inputs.

CSV format (one folder, same as power_system_data_time.py)
----------
    nodes.csv          node, zone[, v_base_kv]
    lines.csv          from, to, capacity, susceptance
    generators.csv     generator, node, p_max, p_min, cost, up_cost, down_cost
    loads.csv          load, node, p_max, p_min
    renewables.csv     renewable, node, capacity, cost, curtailment_cost
    scenarios.csv      scenario, probability
    load_profiles.csv  [time,] scenario, <one column per load>        (MW)
    res_profiles.csv   [time,] scenario, <one column per renewable>   (available MW)
    settings.csv       key, value   (optional: slack_bus, S_BASE_MVA, time_step, gsk_method)

The model is single-period: if the profiles have a `time` column, only the rows of
one time step are used (`time=` argument, default: the first one).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


# Helper functions (pure, independent of the data class)
def _to_dict(M: np.ndarray, rows: list, cols: list) -> dict:
    """2-D array -> nested dict {row_label: {col_label: value}}."""
    return {r: {c: M[i, j].item() for j, c in enumerate(cols)}
            for i, r in enumerate(rows)}


def build_connection_matrices(NODES, GENERATORS, LOADS, ZONES, RENEWABLES,
                              connections_per_node, generators_per_node,
                              loads_per_node, nodes_per_zone,
                              renewables_per_node) -> dict:
    """Adjacency / mapping / incidence matrices (NumPy arrays + line lists)."""
    n_idx = {n: i for i, n in enumerate(NODES)}
    g_idx = {g: i for i, g in enumerate(GENERATORS)}
    d_idx = {d: i for i, d in enumerate(LOADS)}
    z_idx = {z: i for i, z in enumerate(ZONES)}
    r_idx = {r: i for i, r in enumerate(RENEWABLES)}

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

    # node x renewable
    M_nr = np.zeros((len(NODES), len(RENEWABLES)), dtype=int)
    for n, res in renewables_per_node.items():
        for r in res:
            M_nr[n_idx[n], r_idx[r]] = 1

    # zone x node
    M_zn = np.zeros((len(ZONES), len(NODES)), dtype=int)
    for z, nodes in nodes_per_zone.items():
        for n in nodes:
            M_zn[z_idx[z], n_idx[n]] = 1

    # zone-level mappings obtained by chaining
    M_zg = (M_zn @ M_ng > 0).astype(int)
    M_zd = (M_zn @ M_nd > 0).astype(int)
    M_zr = (M_zn @ M_nr > 0).astype(int)
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

    return dict(A_nn=A_nn, M_ng=M_ng, M_nd=M_nd, M_nr=M_nr, M_zn=M_zn, M_zg=M_zg,
                M_zd=M_zd, M_zr=M_zr, A_zz=A_zz, B_nl=B_nl, B_zl=B_zl,
                lines=lines, intra_lines=intra_lines, inter_lines=inter_lines,
                node_zone=node_zone)


def build_gsk(NODES, ZONES, nodes_per_zone, generators_per_node=None,
              gen_max_capacity=None, loads_per_node=None,
              expected_DA_load=None, method: str = 'generation',
              renewables_per_node=None, res_max_capacity=None,
              dispatch=None) -> np.ndarray:
    """Generation shift key GSK[node, zone]; every column sums to 1.

    'generation' weights each node by its installed capacity: conventional
    generators plus renewables (if renewables_per_node / res_max_capacity given).
    'online'     pro rata of what is online: each node is weighted by the output of
                 its units that are running. dispatch = {unit: MW} for generators and
                 renewables (RES after curtailment), e.g. from the base case.
    """
    n_idx = {n: i for i, n in enumerate(NODES)}
    GSK = np.zeros((len(NODES), len(ZONES)))

    if method == 'flat':
        w = {n: 1.0 for n in NODES}
    elif method == 'generation':
        w = {n: sum(gen_max_capacity[g] for g in generators_per_node.get(n, []))
                + sum(res_max_capacity[r] for r in (renewables_per_node or {}).get(n, []))
             for n in NODES}
    elif method == 'load':
        w = {n: sum(expected_DA_load[d] for d in loads_per_node[n])
             for n in NODES}
    elif method == 'online':
        assert dispatch is not None, "GSK method 'online' needs dispatch = {unit: MW}"
        units = lambda n: (generators_per_node.get(n, [])
                           + (renewables_per_node or {}).get(n, []))
        w = {n: sum(max(dispatch.get(u, 0.0), 0.0) for u in units(n))   # offline units weigh 0
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
    N_RES: int = 2

    # sets 
    GENERATORS: list | None = None
    LOADS: list | None = None
    NODES: list | None = None
    ZONES: list | None = None
    SCENARIOS: list | None = None
    RENEWABLES: list | None = None 

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
    renewables_per_node: dict | None = None

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

    # renewables
    res_max_capacity: dict | None = None        # installed MW
    res_cost_coeff: dict | None = None          # EUR/MWh produced (~0)
    res_curtailment_cost: dict | None = None    # EUR/MWh available but not used
    res_scenarios: dict | None = None           # {r: {s: available MW}}
    expected_DA_res: dict | None = None         # DA forecast = expectation over scenarios

    # network parameters 
    intra_zone_capacity: float = 30.0          # line N1-N2
    inter_zone_capacity: float = 100.0         # rest of lines
    intra_zone_susceptance: float = 2.5
    inter_zone_susceptance: float = 2.5
    line_capacity_per_line: dict | None = None     # {(n, m): cap}   override
    line_susceptance_per_line: dict | None = None  # {(n, m): b}     override

    # GSK 
    gsk_method: str = 'generation'            # 'flat' | 'generation' | 'load' | 'online'
    gsk_dispatch: dict | None = None          # {unit: MW}, needed for gsk_method='online'

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
        self._default('RENEWABLES', lambda: [f'R{i+1}' for i in range(self.N_RES)])

        self.indexes = dict(GENERATORS=self.GENERATORS, 
                            LOADS=self.LOADS,
                            NODES=self.NODES, 
                            ZONES=self.ZONES,
                            SCENARIOS=self.SCENARIOS,
                            RENEWABLES=self.RENEWABLES)

        self._default('V_BASE_KV', lambda: {n: 230.0 for n in self.NODES})
        self._default('slack_bus', lambda: self.NODES[0])

        # topology: Zone 1 = {N1, N2}, Zone 2 = {N3}, Zone 3 = {N4}
        self._default('nodes_per_zone',
                      lambda: {'Z1': ['N1', 'N2'], 'Z2': ['N3'], 'Z3': ['N4']})
        self._default('connections_per_node', lambda: {
            'N1': ['N2', 'N3', 'N4'],
            'N2': ['N1','N3'],
            'N3': ['N1', 'N2','N4'],
            'N4': ['N1', 'N3'],
        })
        self._default('generators_per_node', lambda: {
            'N1': ['G1', 'G2'], 'N2': ['G3'], 'N3': ['G4'], 'N4': ['G5'],
        })
        self._default('loads_per_node', lambda: {
            'N1': ['D1'], 'N2': ['D2'], 'N3': ['D3'], 'N4': ['D4'],
        })
        self._default('renewables_per_node', lambda: {
            'N1': ['R1'], 'N2': [], 'N3': [], 'N4': ['R2'],
        })

        # generators: "cap @ cost ± deviation cost"
        # G1 nuclear (N1), G2 CCGT (N1), G3 CCGT (N2), G4 OCGT peaker (N3), G5 coal (N4)
        self._default('gen_linear_cost_coeff',
                      lambda: {'G1': 12, 'G2': 60, 'G3': 65, 'G4': 100, 'G5': 45})
        self._default('gen_down_deviation_cost_coeff',
                      lambda: {'G1': 2, 'G2': 4, 'G3': 4, 'G4': 5, 'G5': 3})
        self._default('gen_up_deviation_cost_coeff',
                      lambda: {'G1': 8, 'G2': 6, 'G3': 6, 'G4': 8, 'G5': 6})
        self._default('gen_max_capacity',
                      lambda: {'G1': 200, 'G2': 150, 'G3': 120, 'G4': 150, 'G5': 200})
        self._default('gen_min_capacity', lambda: {g: 0 for g in self.GENERATORS})

        # loads: D1 city (N1), D2 rural (N2), D3 industry (N3), D4 city (N4); S1 windy, S2 calm
        self._default('load_max_capacity',
                      lambda: {'D1': 300, 'D2': 120, 'D3': 260, 'D4': 200})
        self._default('load_min_capacity', lambda: {d: 0 for d in self.LOADS})
        self._default('load_scenarios', lambda: {
               'D1': {'S1': 50, 'S2': 240},
               'D2': {'S1': 100,  'S2': 90},
               'D3': {'S1': 150, 'S2': 210},
               'D4': {'S1': 150, 'S2': 160}})

        self._default('scenario_probability', lambda: {'S1': 1, 'S2': 0})
        self._default('expected_DA_load', lambda: {
            l: sum(self.scenario_probability[s] * self.load_scenarios[l][s]
                   for s in self.SCENARIOS)
            for l in self.LOADS
        })

        # renewables
        # R1 wind farm (N2), R2 solar park (N4)
        self._default('res_max_capacity', lambda: {'R1': 300, 'R2': 150})
        self._default('res_cost_coeff',   lambda: {'R1': 0.12, 'R2': 0.1})
        self._default('res_curtailment_cost', lambda: {r: 10 for r in self.RENEWABLES})
        self._default('res_scenarios', lambda: {
            'R1': {'S1': 210, 'S2': 60},
            'R2': {'S1': 60,  'S2': 90},
        })
        self._default('expected_DA_res', lambda: {
            r: sum(self.scenario_probability[s] * self.res_scenarios[r][s]
                   for s in self.SCENARIOS)
            for r in self.RENEWABLES
        })

    def _build_mappings(self):
        m = build_connection_matrices(
            self.NODES, self.GENERATORS, self.LOADS, self.ZONES,self.RENEWABLES,
            self.connections_per_node, self.generators_per_node,
            self.loads_per_node, self.nodes_per_zone,
            self.renewables_per_node)
        self.mats = m

        # index lookups (handy for building numpy / solver variables)
        self.node_idx = {n: i for i, n in enumerate(self.NODES)}
        self.gen_idx = {g: i for i, g in enumerate(self.GENERATORS)}
        self.load_idx = {d: i for i, d in enumerate(self.LOADS)}
        self.zone_idx = {z: i for i, z in enumerate(self.ZONES)}
        self.scenario_idx = {s: i for i, s in enumerate(self.SCENARIOS)}
        self.res_idx = {r: i for i, r in enumerate(self.RENEWABLES)}

        self.node_zone = m['node_zone']                      # node -> zone

        # arrays
        self.A_nn, self.M_ng, self.M_nd = m['A_nn'], m['M_ng'], m['M_nd']
        self.M_zn, self.M_zg, self.M_zd = m['M_zn'], m['M_zg'], m['M_zd']
        self.M_nr, self.M_zr = m['M_nr'], m['M_zr']
        self.A_zz, self.B_nl, self.B_zl = m['A_zz'], m['B_nl'], m['B_zl']

        # nested-dict versions
        N, G, D, Z, R = self.NODES, self.GENERATORS, self.LOADS, self.ZONES, self.RENEWABLES
        self.lines = m['lines']
        self.intra_lines = m['intra_lines']
        self.inter_lines = m['inter_lines']
        self.mapping_buses = _to_dict(self.A_nn, N, N)
        self.mapping_generators = _to_dict(self.M_ng, N, G)
        self.mapping_loads = _to_dict(self.M_nd, N, D)
        self.mapping_renewables = _to_dict(self.M_nr, N, R)
        self.mapping_zones = _to_dict(self.M_zn, Z, N) # m matrix 
        self.mapping_zone_generators = _to_dict(self.M_zg, Z, G)
        self.mapping_zone_loads = _to_dict(self.M_zd, Z, D)
        self.mapping_zone_renewables = _to_dict(self.M_zr, Z, R)
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
        self.node_of_renewable = {r: n for n, rs in self.renewables_per_node.items() for r in rs}

    def _build_network(self):
        # per-line capacity and susceptance (ordered like self.lines)
        def is_inter(l):
            return self.node_zone[l[0]] != self.node_zone[l[1]]

        cap = {l: (self.inter_zone_capacity if is_inter(l) else self.intra_zone_capacity)
               for l in self.lines}
        sus = {l: (self.inter_zone_susceptance if is_inter(l) else self.intra_zone_susceptance)
               for l in self.lines}
        def oriented(l):          # (n, m) -> the orientation used in self.lines
            return l if l in cap else (l[1], l[0])

        for l, c in (self.line_capacity_per_line or {}).items():
            cap[oriented(l)] = c
        for l, b in (self.line_susceptance_per_line or {}).items():
            sus[oriented(l)] = b
        assert set(cap) == set(self.lines), "line data refers to a line that is not in the topology"
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
                             method=self.gsk_method,
                             renewables_per_node=self.renewables_per_node,
                             res_max_capacity=self.res_max_capacity,
                             dispatch=self.gsk_dispatch)
        self.gsk = _to_dict(self.GSK, self.NODES, self.ZONES)

        self.PTDF, self.B_bus = build_nodal_ptdf(
            self.NODES, self.B_nl, self.line_b_vec, self.slack_bus)
        
        self.PTDF_zonal = self.PTDF @ self.GSK

        self.bus_susceptance = _to_dict(self.B_bus, self.NODES, self.NODES)
        self.ptdf = _to_dict(self.PTDF, self.lines, self.NODES)
        self.ptdf_zonal = _to_dict(self.PTDF_zonal, self.lines, self.ZONES)

    def _build_vectors(self):
        G, D, S, R = self.GENERATORS, self.LOADS, self.SCENARIOS,self.RENEWABLES
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
        self.res_max_vec = np.array([self.res_max_capacity[r] for r in R], float)
        self.res_cost_vec = np.array([self.res_cost_coeff[r] for r in R], float)
        self.res_curtailment_cost_vec = np.array([self.res_curtailment_cost[r] for r in R], float)
        self.expected_DA_res_vec = np.array([self.expected_DA_res[r] for r in R], float)
        self.res_scenarios_mat = np.array(
            [[self.res_scenarios[r][s] for s in S] for r in R], float)

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
        for r in self.RENEWABLES:
            assert r in self.node_of_renewable, f"renewable {r} is not placed at any node"
            for s in self.SCENARIOS:
                assert 0 <= self.res_scenarios[r][s] <= self.res_max_capacity[r], \
                    f"renewable {r} in scenario {s} exceeds installed capacity"
        # B_bus from the incidence matrix must equal the Laplacian built from line data
        assert np.allclose(self.B_bus.sum(axis=1), 0), "B_bus rows must sum to zero"
        assert np.allclose(self.GSK.sum(axis=0), 1.0), "GSK columns must sum to 1"

    # -------------------------------------------------------------------------
    def summary(self):
        print(f"Nodes      : {self.NODES}  (slack: {self.slack_bus})")
        print(f"Zones      : {self.nodes_per_zone}")
        print(f"Generators : {self.generators_per_node}")
        print(f"Loads      : {self.loads_per_node}")
        print(f"Renewables : {self.renewables_per_node}")
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

    # -------------------------------------------------------------------------
    # CSV input / output
    @classmethod
    def from_csv(cls, folder, time=None, **overrides) -> "PowerSystemData":
        """Build the system from the CSV files in `folder` (format: module docstring).

        time: time step to read from the profiles (only if they have a `time` column);
              default: the first one.
        Any keyword in `overrides` replaces the value read from the files,
        e.g. PowerSystemData.from_csv(path, gsk_method='flat').
        """
        folder = Path(folder)
        read = lambda name: pd.read_csv(folder / name, skipinitialspace=True)

        nodes = read('nodes.csv')
        lines = read('lines.csv')
        gens = read('generators.csv')
        loads = read('loads.csv')
        res = read('renewables.csv')
        scen = read('scenarios.csv')
        load_prof = read('load_profiles.csv')
        res_prof = read('res_profiles.csv')

        NODES = nodes['node'].astype(str).tolist()
        ZONES = list(dict.fromkeys(nodes['zone'].astype(str)))       # order of appearance
        GENERATORS = gens['generator'].astype(str).tolist()
        LOADS = loads['load'].astype(str).tolist()
        RENEWABLES = res['renewable'].astype(str).tolist()
        SCENARIOS = scen['scenario'].astype(str).tolist()

        def per_node(df, key):
            out = {n: [] for n in NODES}
            for k, n in zip(df[key].astype(str), df['node'].astype(str)):
                out[n].append(k)
            return out

        def col(df, key, name):
            return dict(zip(df[key].astype(str), df[name].astype(float)))

        def profiles(df, items, label):
            if 'time' in df:
                t = df['time'].iloc[0] if time is None else time
                df = df[df['time'].astype(str) == str(t)]
                assert len(df), f"{label}: no rows for time '{t}'"
            df = df.assign(scenario=df['scenario'].astype(str)).set_index('scenario')
            missing = [k for k in items if k not in df.columns]
            assert not missing, f"{label}: no column for {missing}"
            assert set(df.index) == set(SCENARIOS) and not df.index.duplicated().any(), \
                f"{label}: need exactly one row per scenario"
            return {k: {s: float(df.at[s, k]) for s in SCENARIOS} for k in items}

        connections = {n: [] for n in NODES}
        line_cap, line_sus = {}, {}
        for row in lines.itertuples(index=False):
            n, m = str(row[0]), str(row[1])
            connections[n].append(m)
            connections[m].append(n)
            line_cap[(n, m)] = float(row.capacity)
            line_sus[(n, m)] = float(row.susceptance)

        kwargs = dict(
            N_GENERATORS=len(GENERATORS), N_LOADS=len(LOADS), N_NODES=len(NODES),
            N_ZONES=len(ZONES), N_SCENARIOS=len(SCENARIOS), N_RES=len(RENEWABLES),
            GENERATORS=GENERATORS, LOADS=LOADS, NODES=NODES, ZONES=ZONES,
            SCENARIOS=SCENARIOS, RENEWABLES=RENEWABLES,
            nodes_per_zone={z: [n for n, zz in zip(NODES, nodes['zone'].astype(str)) if zz == z]
                            for z in ZONES},
            connections_per_node=connections,
            generators_per_node=per_node(gens, 'generator'),
            loads_per_node=per_node(loads, 'load'),
            renewables_per_node=per_node(res, 'renewable'),
            gen_linear_cost_coeff=col(gens, 'generator', 'cost'),
            gen_up_deviation_cost_coeff=col(gens, 'generator', 'up_cost'),
            gen_down_deviation_cost_coeff=col(gens, 'generator', 'down_cost'),
            gen_max_capacity=col(gens, 'generator', 'p_max'),
            gen_min_capacity=col(gens, 'generator', 'p_min'),
            load_max_capacity=col(loads, 'load', 'p_max'),
            load_min_capacity=col(loads, 'load', 'p_min'),
            scenario_probability=col(scen, 'scenario', 'probability'),
            load_scenarios=profiles(load_prof, LOADS, 'load_profiles.csv'),
            res_max_capacity=col(res, 'renewable', 'capacity'),
            res_cost_coeff=col(res, 'renewable', 'cost'),
            res_curtailment_cost=col(res, 'renewable', 'curtailment_cost'),
            res_scenarios=profiles(res_prof, RENEWABLES, 'res_profiles.csv'),
            line_capacity_per_line=line_cap,
            line_susceptance_per_line=line_sus,
        )
        if 'v_base_kv' in nodes:
            kwargs['V_BASE_KV'] = col(nodes, 'node', 'v_base_kv')

        if (folder / 'settings.csv').exists():
            settings = dict(zip(*read('settings.csv')[['key', 'value']].values.T))
            casts = dict(slack_bus=str, gsk_method=str, S_BASE_MVA=float, time_step=float)
            for k, v in settings.items():
                assert k in casts, f"settings.csv: unknown key '{k}'"
                kwargs[k] = casts[k](v)

        kwargs.update(overrides)
        return cls(**kwargs)

    def to_csv(self, folder, time='T1'):
        """Write the current data in the `from_csv` format (useful as a template).

        Profiles get a `time` column (value `time`) so the folder is also readable
        by PowerSystemDataTime.from_csv.
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        pd.DataFrame({'node': self.NODES,
                      'zone': [self.node_zone[n] for n in self.NODES],
                      'v_base_kv': [self.V_BASE_KV[n] for n in self.NODES]}
                     ).to_csv(folder / 'nodes.csv', index=False)
        pd.DataFrame({'from': [l[0] for l in self.lines], 'to': [l[1] for l in self.lines],
                      'capacity': self.line_cap_vec, 'susceptance': self.line_b_vec}
                     ).to_csv(folder / 'lines.csv', index=False)
        pd.DataFrame({'generator': self.GENERATORS,
                      'node': [self.node_of_generator[g] for g in self.GENERATORS],
                      'p_max': self.gen_max_vec, 'p_min': self.gen_min_vec,
                      'cost': self.gen_cost_vec, 'up_cost': self.gen_up_cost_vec,
                      'down_cost': self.gen_down_cost_vec}
                     ).to_csv(folder / 'generators.csv', index=False)
        pd.DataFrame({'load': self.LOADS,
                      'node': [self.node_of_load[d] for d in self.LOADS],
                      'p_max': self.load_max_vec, 'p_min': self.load_min_vec}
                     ).to_csv(folder / 'loads.csv', index=False)
        pd.DataFrame({'renewable': self.RENEWABLES,
                      'node': [self.node_of_renewable[r] for r in self.RENEWABLES],
                      'capacity': self.res_max_vec, 'cost': self.res_cost_vec,
                      'curtailment_cost': self.res_curtailment_cost_vec}
                     ).to_csv(folder / 'renewables.csv', index=False)
        pd.DataFrame({'scenario': self.SCENARIOS, 'probability': self.scenario_prob_vec}
                     ).to_csv(folder / 'scenarios.csv', index=False)

        def profile_frame(items, values):
            return pd.DataFrame([{'time': time, 'scenario': s, **{k: values[k][s] for k in items}}
                                 for s in self.SCENARIOS])

        profile_frame(self.LOADS, self.load_scenarios).to_csv(folder / 'load_profiles.csv', index=False)
        profile_frame(self.RENEWABLES, self.res_scenarios).to_csv(folder / 'res_profiles.csv', index=False)
        pd.DataFrame({'key': ['slack_bus', 'S_BASE_MVA', 'time_step', 'gsk_method'],
                      'value': [self.slack_bus, self.S_BASE_MVA, self.time_step, self.gsk_method]}
                     ).to_csv(folder / 'settings.csv', index=False)


# =============================================================================
if __name__ == "__main__":
    data = PowerSystemData(slack_bus='N4')
    data.summary()
