"""
market_model.py

Market-clearing models built on PowerSystemData:
    1. solve_base_case      nodal DC-OPF (reference dispatch, flows, LMPs)
    2. select_cne           critical network elements for the flow-based domain
       build_flow_based     flow-based domain referenced on the base case (zPTDF, RAM)
    3. da_zonal_clearing    zonal day-ahead market with chance constraints on generators
    4. solve_redispatch     cost-based nodal redispatch of the DA schedule

Usage
-----
    from power_system_data import PowerSystemData
    from market_model import (solve_base_case, select_cne, gsk_online,
                              build_flow_based, da_zonal_clearing, solve_redispatch)

    data = PowerSystemData.from_csv('../data/realistic')
    bc = solve_base_case(data)
    CNE = select_cne(data, 0.05)
    parameters = build_flow_based(data, CNE, bc, gsk=gsk_online(data, bc))
    da = da_zonal_clearing(data, bc, parameters=parameters, sigma_rel=0.1)
    rd = solve_redispatch(data, da, seed=0)
"""

from statistics import NormalDist

import numpy as np
import pandas as pd
import cvxpy as cp

from power_system_data import PowerSystemData, build_gsk


# =============================================================================
# 1. Base case
def solve_base_case(data: PowerSystemData, load=None, res=None, solver=cp.GUROBI, verbose=False):
    """Day-ahead nodal dispatch (DC-OPF via PTDF) at a given load and renewable level.

    load: vector over data.LOADS;      defaults to data.expected_DA_load_vec.
    res:  vector over data.RENEWABLES; defaults to data.expected_DA_res_vec.
    """
    d = data.expected_DA_load_vec if load is None else np.asarray(load, float)
    r = data.expected_DA_res_vec if res is None else np.asarray(res, float)

    g = cp.Variable(len(data.GENERATORS), nonneg=True)      # generator output
    C = cp.Variable(len(data.RENEWABLES), nonneg=True)      # renewable curtailment
    I = cp.Variable(len(data.NODES))                        # nodal net injection
    f = cp.Variable(len(data.lines))                        # line flows

    nodal_balance = I == data.M_ng @ g + data.M_nr @ (r - C) - data.M_nd @ d   # G + r - C - d = P
    cons = [nodal_balance,
            f == data.PTDF @ I,
            cp.sum(I) == 0,
            g >= data.gen_min_vec, g <= data.gen_max_vec,
            C <= r,
            f <= data.line_cap_vec, f >= -data.line_cap_vec]
    prob = cp.Problem(cp.Minimize(data.gen_cost_vec @ g
                                  + data.res_curtailment_cost_vec @ C), cons)
    prob.solve(solver=solver, verbose=verbose)
    assert prob.status == cp.OPTIMAL, f"base case: {prob.status}"

    I_bc = I.value
    return dict(g=pd.Series(g.value, index=data.GENERATORS),
                C=pd.Series(C.value, index=data.RENEWABLES),
                P=pd.Series(I_bc, index=data.NODES),
                f=pd.Series(f.value, index=data.lines),
                NP=pd.Series(data.M_zn @ I_bc, index=data.ZONES),            # zonal net position
                LMP=pd.Series(nodal_balance.dual_value, index=data.NODES),   # nodal price
                cost=prob.value,
                status=prob.status)


# =============================================================================
# 2. Flow-based domain
def select_cne(data: PowerSystemData, threshold=0.05):
    """CNEs = all inter-zonal lines + internal lines with max z2z PTDF >= threshold."""
    Z = len(data.ZONES)
    z2z = np.array([np.abs(data.PTDF_zonal[:, a] - data.PTDF_zonal[:, b])
                    for a in range(Z) for b in range(Z) if a < b]).max(axis=0)   # one value per line
    return [l for k, l in enumerate(data.lines)
            if l in data.inter_lines or z2z[k] >= threshold]


def gsk_online(data: PowerSystemData, bc, res=None):
    """GSK pro rata of what is online in the base case: generator output + RES after curtailment."""
    r = data.expected_DA_res_vec if res is None else np.asarray(res, float)
    dispatch = {**bc["g"].to_dict(), **dict(zip(data.RENEWABLES, r - bc["C"].values))}   # {unit: MW}
    return build_gsk(data.NODES, data.ZONES, data.nodes_per_zone, data.generators_per_node,
                     renewables_per_node=data.renewables_per_node,
                     method='online', dispatch=dispatch)



def build_flow_based(data: PowerSystemData, cne_lines, bc, frm=0, gsk=None, min_ram=None):
    """Flow-based domain referenced on the base case: zPTDF @ NP <= RAM (both directions of each CNE).

    F0  = F_ref - PTDF_z @ NP_ref           flow at zero net positions (loop flows)
    RAM = Fmax - FRM - F0  (+ direction),   Fmax - FRM + F0  (- direction)
    frm: reliability margin as a share of Fmax.
    gsk: defaults to the online GSK of the base case; pass data.GSK for the installed-capacity GSK.
    min_ram: share of Fmax always given to the market (CEP 70 % rule: min_ram=0.7);
             RAM = max(min_ram * Fmax, physical RAM) per direction. None -> physical RAM only.
    """
    L = len(data.lines)
    idx = [data.lines.index(l) for l in cne_lines]
    M = np.eye(L)[idx]                         # K x L  : CNE selection
    K = len(idx)

    GSK = gsk_online(data, bc) if gsk is None else gsk
    PTDF_z = data.PTDF @ GSK                   # L x Z

    Fmax = M @ data.line_cap_vec               # K     : capacity of each CNE
    F_ref = M @ bc["f"].values                 # K     : BC flow on each CNE
    NP_ref = bc["NP"].values                   # Z     : BC zonal net positions
    F0 = F_ref - M @ PTDF_z @ NP_ref           # K     : flow at zero net positions

    S = np.vstack([np.eye(K), -np.eye(K)])     # 2K x K : direction
    RAM = np.tile((1 - frm) * Fmax, 2) - S @ F0
    if min_ram is not None:
        RAM = np.maximum(RAM, np.tile(min_ram * Fmax, 2))
    zPTDF = S @ M @ PTDF_z                     # 2K x Z

    return dict(zPTDF=zPTDF, cne_lines=cne_lines, RAM=RAM, M=M, S=S,
                GSK=GSK, F0=F0, F_ref=F_ref, NP_ref=NP_ref)


# =============================================================================
# 3. Zonal day-ahead market with chance constraints
def da_zonal_clearing(data: PowerSystemData, bc, parameters, load=None, res=None,
                      sigma_rel=0.10, Sigma=None, eps=0.05, solver=cp.GUROBI, verbose=False):
    """Zonal day-ahead clearing on the flow-based domain `parameters` (from build_flow_based).

    RES forecast error omega ~ N(0, Sigma), balanced by generators through participation
    factors alpha; generator limits hold with probability 1 - eps (chance constraints).
    Sigma: defaults to diag((sigma_rel * r)^2).
    """
    d = data.expected_DA_load_vec if load is None else np.asarray(load, float)
    r = data.expected_DA_res_vec if res is None else np.asarray(res, float)

    NP_da = cp.Variable(len(data.ZONES))
    G_da = cp.Variable(len(data.GENERATORS), nonneg=True)
    C_da = cp.Variable(len(data.RENEWABLES), nonneg=True)
    alpha = cp.Variable(len(data.GENERATORS), nonneg=True)    # participation factors

    if Sigma is None:
        Sigma = np.diag((sigma_rel * r) ** 2)
    Sh = np.linalg.cholesky(Sigma + 1e-12 * np.eye(len(data.RENEWABLES)))   # Sigma = Sh Sh^T
    sig_s = np.linalg.norm(Sh.T @ np.ones(data.N_RES))                     # std of e^T omega
    z_gen = NormalDist().inv_cdf(1 - eps)

    balance = NP_da == data.M_zg @ G_da + data.M_zr @ (r - C_da) - data.M_zd @ d
    cons = [balance, C_da <= r,
            G_da + z_gen * sig_s * alpha <= data.gen_max_vec,     # [CC] upper
            G_da - z_gen * sig_s * alpha >= 0,                    # [CC] lower
            parameters["zPTDF"] @ NP_da <= parameters["RAM"],
            cp.sum(NP_da) == 0,
            cp.sum(alpha) == 1]

    prob = cp.Problem(cp.Minimize(data.gen_cost_vec @ G_da
                                  + data.res_curtailment_cost_vec @ C_da), cons)
    prob.solve(solver=solver, verbose=verbose)
    assert prob.status == cp.OPTIMAL, f"day-ahead: {prob.status}"

    return dict(G=pd.Series(G_da.value, index=data.GENERATORS),
                C=pd.Series(C_da.value, index=data.RENEWABLES),
                NP=pd.Series(NP_da.value, index=data.ZONES),
                alpha=pd.Series(alpha.value, index=data.GENERATORS),
                price=pd.Series(balance.dual_value, index=data.ZONES),   # zonal price
                Sigma=Sigma,                                             # forecast-error covariance of the CC
                r_forecast=pd.Series(r, index=data.RENEWABLES),          # mean of the CC distribution
                cost=prob.value)


# =============================================================================
# 4. Redispatch
def solve_redispatch(data: PowerSystemData, da, load=None, res=None, seed=None, margin=0.95,
                     solver=cp.GUROBI, verbose=False):
    """Cost-based redispatch after the zonal DA: adjust the DA schedule at nodal level so that
    every line respects margin * its thermal limit (PTDF flows, no RAM / GSK approximation).

    up/down cost per MW:  up   -> paid  c_g + c_up   (generator produces more)
                          down -> refund c_g - c_down (generator saves its fuel, keeps a premium)
    so DA cost + redispatch cost = cost of the final dispatch + deviation premiums.

    res: RES availability in real time. If None, one realization is drawn from the same
         distribution as the DA chance constraint: r = r_forecast + omega, omega ~ N(0, Sigma),
         clipped to [0, installed capacity]. seed makes the draw reproducible.
    """
    d = data.expected_DA_load_vec if load is None else np.asarray(load, float)
    if res is None:
        rng = np.random.default_rng(seed)
        omega = rng.multivariate_normal(np.zeros(len(data.RENEWABLES)), da["Sigma"])   # forecast error
        r = np.clip(da["r_forecast"].values + omega, 0, data.res_max_vec)
    else:
        r = np.asarray(res, float)

    G_da = da["G"].values      # DA schedule
    C_da = da["C"].values      # DA curtailment

    up = cp.Variable(len(data.GENERATORS), nonneg=True)     # upward redispatch
    down = cp.Variable(len(data.GENERATORS), nonneg=True)   # downward redispatch
    dC = cp.Variable(len(data.RENEWABLES))                  # change in curtailment (+ = curtail more)
    f = cp.Variable(len(data.lines))                        # line flows after redispatch

    G_rd = G_da + up - down
    C_rd = C_da + dC
    P_rd = data.M_ng @ G_rd + data.M_nr @ (r - C_rd) - data.M_nd @ d   # nodal injection after redispatch

    cons = [f == data.PTDF @ P_rd,
            cp.sum(P_rd) == 0,                              # redispatch is balanced
            G_rd >= data.gen_min_vec, G_rd <= data.gen_max_vec,
            C_rd >= 0, C_rd <= r,
            f <= margin * data.line_cap_vec, f >= -margin * data.line_cap_vec]

    rd_cost = ((data.gen_cost_vec + data.gen_up_cost_vec) @ up
               - (data.gen_cost_vec - data.gen_down_cost_vec) @ down
               + data.res_curtailment_cost_vec @ dC)
    prob = cp.Problem(cp.Minimize(rd_cost), cons)
    prob.solve(solver=solver, verbose=verbose)
    assert prob.status == cp.OPTIMAL, f"redispatch: {prob.status}"

    return dict(G=pd.Series(G_rd.value, index=data.GENERATORS),
                C=pd.Series(C_rd.value, index=data.RENEWABLES),
                up=pd.Series(up.value, index=data.GENERATORS),
                down=pd.Series(down.value, index=data.GENERATORS),
                dC=pd.Series(dC.value, index=data.RENEWABLES),
                P=pd.Series(P_rd.value, index=data.NODES),
                f=pd.Series(f.value, index=data.lines),
                r=pd.Series(np.asarray(r, float), index=data.RENEWABLES),   # RES availability used
                volume=up.value.sum() + down.value.sum() + np.abs(dC.value).sum(),
                cost=prob.value)


# =============================================================================
if __name__ == "__main__":
    data = PowerSystemData()
    bc = solve_base_case(data)
    CNE = select_cne(data, 0.05)
    parameters = build_flow_based(data, CNE, bc)
    da = da_zonal_clearing(data, bc, parameters=parameters)
    rd = solve_redispatch(data, da, seed=0)
    print(f"base case {bc['cost']:.1f}  |  DA {da['cost']:.1f}  |  RD {rd['cost']:.1f} "
          f"(volume {rd['volume']:.1f} MW)  |  DA + RD {da['cost'] + rd['cost']:.1f}")
