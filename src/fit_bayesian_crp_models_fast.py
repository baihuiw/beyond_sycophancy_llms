from __future__ import annotations

import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

SCALE = np.arange(1, 8, dtype=float)
EPS = 1e-12
OPT_N_RANDOM = 450
OPT_N_COORD = 80


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)



def find_file(data_dir: Path, candidates: List[str]) -> Optional[Path]:
    """Return first existing file among common filename variants."""
    if data_dir.is_file():
        if data_dir.name in candidates or any(c in data_dir.name for c in candidates):
            return data_dir
    for name in candidates:
        p = data_dir / name
        if p.exists():
            return p
    # fuzzy fallback: useful for files like study1_results(1).jsonl
    for p in data_dir.glob("*.jsonl"):
        for name in candidates:
            stem = name.replace(".jsonl", "")
            if stem in p.name:
                return p
    return None


def pvec(d: Dict[str, float]) -> np.ndarray:
    v = np.array([float(d.get(str(i), 0.0) or 0.0) for i in range(1, 8)], dtype=float)
    v = np.clip(v, EPS, None)
    return v / v.sum()


def softmax_mat(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return np.clip(e / e.sum(axis=1, keepdims=True), EPS, 1.0)


def ce(y: np.ndarray, q: np.ndarray) -> float:
    return float(-np.mean(np.sum(y * np.log(np.clip(q, EPS, 1.0)), axis=1)))


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    den = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - yhat) ** 2) / den) if den > 0 else np.nan


def K(pos: np.ndarray) -> np.ndarray:
    """N x 7 absolute-distance kernel centered on pos."""
    pos = np.asarray(pos, dtype=float).reshape(-1, 1)
    return -np.abs(SCALE.reshape(1, -1) - pos)


def _clip_to_bounds(x: np.ndarray, bounds):
    lo = np.array([b[0] if b[0] is not None else -np.inf for b in bounds], dtype=float)
    hi = np.array([b[1] if b[1] is not None else np.inf for b in bounds], dtype=float)
    return np.minimum(np.maximum(x, lo), hi)


def no_scipy_minimize(obj, init, bounds, seed=13, n_random=None, n_coord=None):
    """Small derivative-free optimizer; avoids scipy.optimize entirely."""
    if n_random is None:
        n_random = OPT_N_RANDOM
    if n_coord is None:
        n_coord = OPT_N_COORD
    rng = np.random.default_rng(seed)
    init = _clip_to_bounds(np.array(init, dtype=float), bounds)
    dim = len(init)

    lo = np.array([b[0] if b[0] is not None else -10 for b in bounds], dtype=float)
    hi = np.array([b[1] if b[1] is not None else 10 for b in bounds], dtype=float)

    candidates = [init]
    # Uniform random starts inside bounded parameter box.
    for _ in range(n_random):
        candidates.append(lo + rng.random(dim) * (hi - lo))

    vals = np.array([obj(c) for c in candidates], dtype=float)
    best = np.array(candidates[int(np.argmin(vals))], dtype=float)
    best_val = float(np.min(vals))

    # Coordinate pattern search with shrinking step sizes.
    step = 0.25 * (hi - lo)
    step = np.where(np.isfinite(step) & (step > 0), step, 1.0)
    for _ in range(n_coord):
        improved = False
        for j in range(dim):
            for sign in (1.0, -1.0):
                cand = best.copy()
                cand[j] += sign * step[j]
                cand = _clip_to_bounds(cand, bounds)
                val = obj(cand)
                if val + 1e-12 < best_val:
                    best, best_val, improved = cand, float(val), True
        if not improved:
            step *= 0.55
        if np.max(step) < 1e-5:
            break
    return best, best_val


def fit(name, y, pred_fn, init, bounds, delta_true=None, delta_pred_fn=None):
    def obj(theta):
        return ce(y, pred_fn(np.array(theta, dtype=float)))
    theta, loss = no_scipy_minimize(obj, init, bounds)
    q = pred_fn(theta)
    out = {"model": name, "n": len(y), "loss_cross_entropy": ce(y, q), "parameters": json.dumps(theta.tolist())}
    if delta_true is not None and delta_pred_fn is not None:
        out["r2_target_metric"] = r2(delta_true, delta_pred_fn(q))
    return out, theta.tolist(), q

def load_baselines(path: Path):
    base = {}
    for o in read_jsonl(path):
        base[o["metadata"]["qid"]] = pvec(o["response"]["probs"])
    return base


def study1(data_dir, base, opposing_only=False):
    rows = []
    study1_path = find_file(data_dir, ["study1_results.jsonl", "study1_results(1).jsonl"])
    if study1_path is None:
        raise FileNotFoundError("Could not find study1_results.jsonl in --data-dir")
    for o in read_jsonl(study1_path):
        md = o["metadata"]
        if md.get("condition") == "memory_only":
            continue
        if opposing_only and "oppose" not in md.get("condition", ""):
            continue
        qid = md["qid"]
        p0 = base[qid]
        mode = int(np.argmax(p0) + 1)
        target = int(md["cue_target"])
        rows.append((p0, pvec(o["response"]["probs"]), mode, target, abs(target - mode)))

    p0 = np.vstack([r[0] for r in rows]); y = np.vstack([r[1] for r in rows])
    mode = np.array([r[2] for r in rows]); target = np.array([r[3] for r in rows]); dist = np.array([r[4] for r in rows], dtype=float)
    logp0 = np.log(np.clip(p0, EPS, 1.0)); kt = K(target); km = K(mode)
    idx = target - 1
    delta_true = y[np.arange(len(y)), idx] - p0[np.arange(len(y)), idx]
    delta_pred = lambda q: q[np.arange(len(q)), idx] - p0[np.arange(len(q)), idx]

    fits, params = [], {}
    specs = {
        "S1 prior only: q ∝ p0^rho": (
            lambda th: softmax_mat(th[0] * logp0), [1.0], [(0, 10)]),
        "S1 constant cue: q ∝ p0^rho exp(lambda K(i,t))": (
            lambda th: softmax_mat(th[0] * logp0 + th[1] * kt), [1, 1], [(0, 10), (-10, 10)]),
        "S1 distance-decay cue: lambda(d)=lambda0 exp(-gamma d)": (
            lambda th: softmax_mat(th[0] * logp0 + (th[1] * np.exp(-th[2] * dist)).reshape(-1,1) * kt), [1, 2, .5], [(0,10),(-10,10),(0,5)]),
        "S1 distance-decay + prior-anchor": (
            lambda th: softmax_mat(th[0] * logp0 + (th[1] * np.exp(-th[2] * dist)).reshape(-1,1) * kt + th[3] * km), [1,2,.5,1], [(0,10),(-10,10),(0,5),(-10,10)]),
    }
    for name, (fn, init, bounds) in specs.items():
        res, par, _ = fit(name, y, fn, init, bounds, delta_true, delta_pred)
        res["study"] = "Study 1"
        fits.append(res); params[name] = par
    return fits, params


def study2(data_dir, base):
    study2_path = find_file(data_dir, ["study2_v3_experimental_outputs.jsonl", "study2_results.jsonl"])
    if study2_path is None:
        raise FileNotFoundError("Could not find study2_v3_experimental_outputs.jsonl in --data-dir")
    raw = list(read_jsonl(study2_path))
    inj_rows, d1_map, corr_rows = [], {}, []
    for o in raw:
        md = o["metadata"]; qid = md["qid"]
        if qid not in base: continue
        cond = md["condition"]; inj = int(md["irrational_choice"]); mode = int(md["mode_choice"])
        y = pvec(o["response"]["probs"]); p0 = base[qid]
        if cond in {"memory_only", "memory_justify"}:
            inj_rows.append((p0, y, mode, inj, abs(inj - mode)))
            if cond == "memory_justify": d1_map[(qid, inj)] = y
    for o in raw:
        md = o["metadata"]; qid = md["qid"]
        if qid not in base or md["condition"] != "memory_justify_correction": continue
        inj = int(md["irrational_choice"]); mode = int(md["mode_choice"])
        if (qid, inj) not in d1_map: continue
        corr_rows.append((base[qid], d1_map[(qid, inj)], pvec(o["response"]["probs"]), mode, inj, abs(inj-mode)))

    fits, params = [], {}
    # Injection
    p0 = np.vstack([r[0] for r in inj_rows]); y = np.vstack([r[1] for r in inj_rows])
    mode = np.array([r[2] for r in inj_rows]); target = np.array([r[3] for r in inj_rows]); dist = np.array([r[4] for r in inj_rows], dtype=float)
    logp0 = np.log(p0); kt = K(target); idx = target - 1
    delta_true = y[np.arange(len(y)), idx] - p0[np.arange(len(y)), idx]
    delta_pred = lambda q: q[np.arange(len(q)), idx] - p0[np.arange(len(q)), idx]
    inj_specs = {
        "S2 injection constant": (lambda th: softmax_mat(th[0]*logp0 + th[1]*kt), [1,1], [(0,10),(-10,10)]),
        "S2 injection distance-decay": (lambda th: softmax_mat(th[0]*logp0 + (th[1]*np.exp(-th[2]*dist)).reshape(-1,1)*kt), [1,2,.5], [(0,10),(-10,10),(0,5)]),
    }
    for name, (fn, init, bounds) in inj_specs.items():
        res, par, _ = fit(name, y, fn, init, bounds, delta_true, delta_pred)
        res["study"] = "Study 2 injection"; fits.append(res); params[name] = par

    # Correction
    p0 = np.vstack([r[0] for r in corr_rows]); d1 = np.vstack([r[1] for r in corr_rows]); y = np.vstack([r[2] for r in corr_rows])
    mode = np.array([r[3] for r in corr_rows]); target = np.array([r[4] for r in corr_rows])
    logp0 = np.log(p0); logd1 = np.log(d1); km = K(mode); kt = K(target); idx = target - 1
    resid_true = y[np.arange(len(y)), idx] - p0[np.arange(len(y)), idx]
    resid_pred = lambda q: q[np.arange(len(q)), idx] - p0[np.arange(len(q)), idx]
    corr_specs = {
        "S2 correction restore baseline": (lambda th: softmax_mat(th[0]*logp0 + th[1]*km), [1,1], [(0,10),(-10,10)]),
        "S2 correction baseline + residual injected": (lambda th: softmax_mat(th[0]*logp0 + th[1]*km + th[2]*kt), [1,1,1], [(0,10),(-10,10),(-10,10)]),
        "S2 correction sequential D1-as-prior": (lambda th: softmax_mat(th[0]*logd1 + th[1]*km), [1,1], [(0,10),(-10,10)]),
    }
    for name, (fn, init, bounds) in corr_specs.items():
        res, par, _ = fit(name, y, fn, init, bounds, resid_true, resid_pred)
        res["study"] = "Study 2 correction"; fits.append(res); params[name] = par
    return fits, params


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def same_side_flag(x: int, m: int) -> float:
    """Whether peer position and A1 mode are on the same side of neutral 4."""
    if x == 4 or m == 4:
        return 0.0
    return float((x < 4 and m < 4) or (x > 4 and m > 4))


def parse_coalition(condition: str) -> str:
    """Map a free-form condition string to one of {3:0, 2:1, 1:2, 0:3, other}.

    Tries common encodings: '3_0', '3:0', 'full_support', 'majority_support', etc.
    Falls back to 'other' so rows are never dropped.
    """
    if condition is None:
        return "other"
    c = str(condition).lower()
    # numeric encodings
    for k in ["3_0", "3:0", "3-0", "30"]:
        if k in c: return "3:0"
    for k in ["2_1", "2:1", "2-1", "21"]:
        if k in c: return "2:1"
    for k in ["1_2", "1:2", "1-2", "12"]:
        if k in c: return "1:2"
    for k in ["0_3", "0:3", "0-3", "03"]:
        if k in c: return "0:3"
    # word encodings
    if "full_support" in c or "fullsupport" in c: return "3:0"
    if "majority_support" in c or "majoritysupport" in c: return "2:1"
    if "majority_oppos" in c or "majorityoppos" in c: return "1:2"
    if "full_oppos" in c or "fulloppos" in c: return "0:3"
    return "other"


COALITION_LEVELS = ["3:0", "2:1", "1:2", "0:3", "other"]


def _row_from_summary(o: dict, model_id: str) -> Optional[Tuple]:
    """Build a Study 3 row tuple from a single summaries-format record.

    Summaries records have one entry per episode with everything pre-aggregated:
      - baseline_dist_probs / final_dist_probs : prob distributions over 1..7
      - peer_positions / peer_roles            : peer state for the episode
      - condition / ratio                      : coalition labels
      - qid                                    : dilemma identifier
    Returns None if the record is malformed (missing distributions or no peers).
    """
    if "baseline_dist_probs" not in o or "final_dist_probs" not in o:
        return None
    p0 = pvec(o["baseline_dist_probs"])
    y = pvec(o["final_dist_probs"])
    mode = int(np.argmax(p0) + 1)
    qid = o.get("qid", "unknown")
    condition = o.get("condition", "unknown")

    # Prefer the explicit ratio field; fall back to parsing condition.
    coalition = o.get("ratio") or parse_coalition(condition)
    if coalition not in COALITION_LEVELS:
        coalition = parse_coalition(condition)

    peers_raw = o.get("peer_positions") or []
    roles_raw = o.get("peer_roles") or []
    peers = [int(x) for x in peers_raw if x is not None]
    roles = [str(r) for r in roles_raw if r is not None]

    # Defensive fallback if roles are missing or misaligned.
    if len(roles) != len(peers):
        roles = []
        for x in peers:
            if x == mode or same_side_flag(x, mode):
                roles.append("supporter")
            else:
                roles.append("opposer")

    if not peers:
        return None
    return (qid, condition, p0, y, mode, peers, roles, coalition, model_id)


def load_summaries_rows(
    summaries_paths: List[Path],
    per_model_id: Optional[str] = None,
) -> List[Tuple]:
    """Load Study 3 rows from one or more summaries-format files.

    Model identity is inferred from each filename's stem: a file named
    summaries_phi.jsonl yields model_id 'phi'; transcripts_phi_4.jsonl yields
    'phi_4'. Callers can override by ensuring filenames are clean.
    """
    rows: List[Tuple] = []
    for path in summaries_paths:
        # Infer model id from filename. Strip leading 'summaries_' or 'transcripts_'.
        stem = path.stem
        for prefix in ("summaries_", "transcripts_", "summary_"):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break
        model_id = stem if stem else "unknown"
        if per_model_id is not None and model_id != per_model_id:
            continue
        for o in read_jsonl(path):
            row = _row_from_summary(o, model_id)
            if row is not None:
                rows.append(row)
    return rows


def load_transcripts_rows(
    data_dir: Path,
    per_model_id: Optional[str] = None,
) -> List[Tuple]:
    """Load Study 3 rows from a transcripts.jsonl file (multi-turn format).

    This reproduces the row-building logic the original v2/v3 study3 used.
    """
    episodes: Dict[Tuple, List[dict]] = {}
    study3_path = find_file(data_dir, ["transcripts.jsonl", "transcripts(1).jsonl"])
    if study3_path is None:
        raise FileNotFoundError("Could not find transcripts.jsonl in --data-dir")
    for o in read_jsonl(study3_path):
        m_id = (
            o.get("model")
            or o.get("model_id")
            or o.get("model_name")
            or (o.get("metadata") or {}).get("model")
            or (o.get("metadata") or {}).get("model_id")
            or "unknown"
        )
        if per_model_id is not None and str(m_id) != per_model_id:
            continue
        o["_model_id"] = str(m_id)
        episodes.setdefault((o["qid"], o["condition"], str(m_id)), []).append(o)

    rows: List[Tuple] = []
    for key, turns in episodes.items():
        turns = sorted(turns, key=lambda x: x.get("turn_index", 0))
        b = next((t for t in turns if t.get("agent") == "A1" and t.get("a1_dist_probs")), None)
        finals = [t for t in turns if t.get("agent") == "A1" and t.get("a1_dist_probs_final")]
        if not b or not finals:
            continue
        f = finals[-1]
        p0 = pvec(b["a1_dist_probs"])
        y = pvec(f["a1_dist_probs_final"])
        mode = int(np.argmax(p0) + 1)
        qid = key[0]
        condition = key[1]
        model_id = key[2]
        coalition = parse_coalition(condition)

        peers = f.get("peer_positions") or [t.get("fixed_position") for t in turns if t.get("agent") != "A1"]
        roles = f.get("peer_roles") or [t.get("agent_role") for t in turns if t.get("agent") != "A1"]
        peers = [int(x) for x in peers if x is not None]
        roles = [str(r) for r in roles if r is not None]

        if len(roles) != len(peers):
            roles = []
            for x in peers:
                if x == mode or same_side_flag(x, mode):
                    roles.append("supporter")
                else:
                    roles.append("opposer")

        if peers:
            rows.append((qid, condition, p0, y, mode, peers, roles, coalition, model_id))
    return rows


def study3(data_dir, per_model_id: Optional[str] = None,
           summaries_paths: Optional[List[Path]] = None):
    """Fit Study 3 social-updating models.

    Source selection:
      - If summaries_paths is provided, load rows from those files via
        load_summaries_rows. Model identity is inferred from filenames.
      - Otherwise, load from transcripts.jsonl in data_dir via
        load_transcripts_rows.

    The fitting code below is identical regardless of source.
    """
    if summaries_paths is not None:
        rows = load_summaries_rows(summaries_paths, per_model_id=per_model_id)
    else:
        rows = load_transcripts_rows(data_dir, per_model_id=per_model_id)

    if not rows:
        raise ValueError(
            "No Study 3 rows produced. "
            "Check --data-dir or --summaries paths and per-model filter."
        )

    p0 = np.vstack([r[2] for r in rows])
    y = np.vstack([r[3] for r in rows])
    mode = np.array([r[4] for r in rows], dtype=int)
    p_mode = p0[np.arange(len(rows)), mode - 1]
    logp0 = np.log(np.clip(p0, EPS, 1.0))
    km = K(mode)
    n = len(rows)

    # Per-row coalition condition (one of {3:0, 2:1, 1:2, 0:3, other}) and model id.
    coalition_idx = np.array(
        [COALITION_LEVELS.index(r[7]) for r in rows], dtype=int
    )
    model_ids = np.array([r[8] for r in rows], dtype=object)

    # Vectorized peer tensors: N x J, where J is usually 3.
    max_peers = max(len(r[5]) for r in rows)
    peer_x = np.zeros((n, max_peers), dtype=float)
    peer_mask = np.zeros((n, max_peers), dtype=float)
    sup_mask = np.zeros((n, max_peers), dtype=float)
    opp_mask = np.zeros((n, max_peers), dtype=float)
    same_side = np.zeros((n, max_peers), dtype=float)
    for a, row in enumerate(rows):
        m = row[4]
        peers = row[5]
        roles = row[6]
        for j, (x, role) in enumerate(zip(peers, roles)):
            peer_x[a, j] = float(x)
            peer_mask[a, j] = 1.0
            role_l = role.lower()
            if "support" in role_l:
                sup_mask[a, j] = 1.0
            elif "oppos" in role_l or "correct" in role_l:
                opp_mask[a, j] = 1.0
            elif x == m or same_side_flag(x, m):
                sup_mask[a, j] = 1.0
            else:
                opp_mask[a, j] = 1.0
            same_side[a, j] = same_side_flag(int(x), int(m))

    dist = np.abs(peer_x - mode.reshape(-1, 1)) * peer_mask
    # kernels[a,j,i] = K(i, peer_j)
    kernels = -np.abs(SCALE.reshape(1, 1, 7) - peer_x[:, :, None]) * peer_mask[:, :, None]

    evidence_const = np.sum(kernels, axis=1)

    idx = mode - 1
    delta_true = y[np.arange(n), idx] - p0[np.arange(n), idx]
    delta_pred = lambda q: q[np.arange(len(q)), idx] - p0[np.arange(len(q)), idx]

    def gated_evidence(gamma):
        weights = np.exp(-gamma * dist) * peer_mask
        return np.sum(weights[:, :, None] * kernels, axis=1)

    def split_gated_evidence(gamma_s, gamma_o):
        w_s = np.exp(-gamma_s * dist) * sup_mask
        w_o = np.exp(-gamma_o * dist) * opp_mask
        ev_s = np.sum(w_s[:, :, None] * kernels, axis=1)
        ev_o = np.sum(w_o[:, :, None] * kernels, axis=1)
        return ev_s, ev_o

    def crp_membership_evidence(a0, b_dist, c_same):
        # w_j = sigmoid(a0 - b_dist * distance + c_same * same_side_j)
        weights = sigmoid(a0 - b_dist * dist + c_same * same_side) * peer_mask
        return np.sum(weights[:, :, None] * kernels, axis=1)

    def pred_split(th):
        ev_s, ev_o = split_gated_evidence(th[2], th[4])
        return softmax_mat(th[0] * logp0 + th[1] * ev_s + th[3] * ev_o)

    def pred_split_cert(th):
        ev_s, ev_o = split_gated_evidence(th[3], th[5])
        rho = (th[0] + th[1] * p_mode).reshape(-1, 1)
        return softmax_mat(rho * logp0 + th[2] * ev_s + th[4] * ev_o)

    def pred_crp(th):
        return softmax_mat(th[0] * logp0 + th[1] * crp_membership_evidence(th[2], th[3], th[4]))

    def pred_crp_cert(th):
        rho = (th[0] + th[1] * p_mode).reshape(-1, 1)
        return softmax_mat(rho * logp0 + th[2] * crp_membership_evidence(th[3], th[4], th[5]))

    # ---- v3 additions ----------------------------------------------------

    def pred_split_per_condition(th):
        """Split supporter/opposer model with per-coalition multipliers on
        supporter and opposer evidence streams.

        th layout (12 params):
          [0]  rho                  (prior weight)
          [1]  lambda_supporter     (base supporter influence)
          [2]  gamma_supporter      (supporter distance decay)
          [3]  lambda_opposer       (base opposer influence)
          [4]  gamma_opposer        (opposer distance decay)
          [5..8]  c_sup[3:0, 2:1, 1:2, 0:3]  per-coalition supporter multipliers
          [9..12] c_opp[3:0, 2:1, 1:2, 0:3]  per-coalition opposer multipliers
        Conditions tagged 'other' get multiplier 1.0 (the base lambda).
        """
        ev_s, ev_o = split_gated_evidence(th[2], th[4])
        c_sup_levels = np.array([th[5], th[6], th[7], th[8], 1.0])
        c_opp_levels = np.array([th[9], th[10], th[11], th[12], 1.0])
        c_sup = c_sup_levels[coalition_idx].reshape(-1, 1)
        c_opp = c_opp_levels[coalition_idx].reshape(-1, 1)
        return softmax_mat(th[0] * logp0 + th[1] * c_sup * ev_s + th[3] * c_opp * ev_o)

    def pred_split_saturating(th):
        """Split supporter/opposer with tanh saturation on each evidence stream.
        Captures diminishing returns from additional aligned peers.

        th layout (7 params):
          [0]  rho
          [1]  lambda_supporter
          [2]  gamma_supporter
          [3]  beta_supporter   (saturation steepness for supporters)
          [4]  lambda_opposer
          [5]  gamma_opposer
          [6]  beta_opposer
        """
        ev_s, ev_o = split_gated_evidence(th[2], th[5])
        # Tanh saturation per-position over the position dimension. We use
        # tanh on the per-row aggregate magnitude, applied position-wise.
        ev_s_sat = np.tanh(th[3] * ev_s)
        ev_o_sat = np.tanh(th[6] * ev_o)
        return softmax_mat(th[0] * logp0 + th[1] * ev_s_sat + th[4] * ev_o_sat)

    # Pre-compute extreme-same-side mask: peer is on same side as A1 mode AND
    # holds a position 5+ steps away (i.e., position 1 or 7 when mode != 4).
    EXTREME_THRESHOLD = 4.0
    extreme_same_side_mask = (dist >= EXTREME_THRESHOLD) * sup_mask  # N x J
    # Sum kernels weighted by extreme-same-side mask (penalty target = each row's mode).
    extreme_evidence = np.sum(extreme_same_side_mask[:, :, None] * kernels, axis=1)

    def pred_split_extreme_reject(th):
        """Split supporter/opposer + explicit extreme-same-side rejection term.

        The extreme-rejection term lets the model push *away* from the focal
        position when a same-side peer is at an extreme location, capturing
        the subtyping behaviour discussed in the paper.

        th layout (6 params):
          [0]  rho
          [1]  lambda_supporter
          [2]  gamma_supporter
          [3]  lambda_opposer
          [4]  gamma_opposer
          [5]  zeta_extreme_reject  (negative values = move away from extreme allies)
        """
        ev_s, ev_o = split_gated_evidence(th[2], th[4])
        return softmax_mat(
            th[0] * logp0
            + th[1] * ev_s
            + th[3] * ev_o
            + th[5] * extreme_evidence
        )

    def pred_full(th):
        """Combined model: per-condition multipliers + tanh saturation +
        extreme-rejection. The most expressive Study 3 model.

        th layout (16 params):
          [0]  rho
          [1]  lambda_supporter
          [2]  gamma_supporter
          [3]  beta_supporter
          [4]  lambda_opposer
          [5]  gamma_opposer
          [6]  beta_opposer
          [7]  zeta_extreme_reject
          [8..11]  c_sup[3:0, 2:1, 1:2, 0:3]
          [12..15] c_opp[3:0, 2:1, 1:2, 0:3]
        """
        ev_s, ev_o = split_gated_evidence(th[2], th[5])
        c_sup_levels = np.array([th[8], th[9], th[10], th[11], 1.0])
        c_opp_levels = np.array([th[12], th[13], th[14], th[15], 1.0])
        c_sup = c_sup_levels[coalition_idx].reshape(-1, 1)
        c_opp = c_opp_levels[coalition_idx].reshape(-1, 1)
        ev_s_sat = np.tanh(th[3] * ev_s)
        ev_o_sat = np.tanh(th[6] * ev_o)
        return softmax_mat(
            th[0] * logp0
            + th[1] * c_sup * ev_s_sat
            + th[4] * c_opp * ev_o_sat
            + th[7] * extreme_evidence
        )

    fits, params = [], {}

    specs = {
        "S3 prior only": (
            lambda th: softmax_mat(th[0] * logp0),
            [1.0],
            [(0, 10)],
            ["rho"],
        ),
        "S3 constant peer pooling": (
            lambda th: softmax_mat(th[0] * logp0 + th[1] * evidence_const),
            [1.0, 1.0],
            [(0, 10), (-10, 10)],
            ["rho", "lambda_peer"],
        ),
        "S3 CRP-like distance-gated pooling": (
            lambda th: softmax_mat(th[0] * logp0 + th[1] * gated_evidence(th[2])),
            [1.0, 2.0, 0.5],
            [(0, 10), (-10, 10), (0, 5)],
            ["rho", "lambda_peer", "gamma_distance"],
        ),
        "S3 CRP-like distance-gated pooling + prior-anchor": (
            lambda th: softmax_mat(th[0] * logp0 + th[1] * gated_evidence(th[2]) + th[3] * km),
            [1.0, 2.0, 0.5, 0.0],
            [(0, 10), (-10, 10), (0, 5), (-10, 10)],
            ["rho", "lambda_peer", "gamma_distance", "eta_mode_anchor"],
        ),
        "S3 split supporter/opposer distance-gated pooling": (
            pred_split,
            [1.0, 2.0, 0.5, 1.0, 0.5],
            [(0, 10), (-10, 10), (0, 5), (-10, 10), (0, 5)],
            ["rho", "lambda_supporter", "gamma_supporter", "lambda_opposer", "gamma_opposer"],
        ),
        "S3 split supporter/opposer + baseline-certainty rho": (
            pred_split_cert,
            [0.5, 1.0, 2.0, 0.5, 1.0, 0.5],
            [(0, 10), (-10, 10), (-10, 10), (0, 5), (-10, 10), (0, 5)],
            ["rho0", "rho_pmode", "lambda_supporter", "gamma_supporter", "lambda_opposer", "gamma_opposer"],
        ),
        "S3 CRP latent in-group probability": (
            pred_crp,
            [1.0, 2.0, 0.0, 1.0, 1.0],
            [(0, 10), (-10, 10), (-6, 6), (0, 6), (-6, 6)],
            ["rho", "lambda_group_evidence", "a0_membership", "b_distance", "c_same_side"],
        ),
        "S3 CRP latent in-group + baseline-certainty rho": (
            pred_crp_cert,
            [0.5, 1.0, 2.0, 0.0, 1.0, 1.0],
            [(0, 10), (-10, 10), (-10, 10), (-6, 6), (0, 6), (-6, 6)],
            ["rho0", "rho_pmode", "lambda_group_evidence", "a0_membership", "b_distance", "c_same_side"],
        ),
        # ---- v3 specs ---------------------------------------------------
        "S3 v3 split + per-coalition multipliers": (
            pred_split_per_condition,
            [1.0, 2.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [(0, 10), (-10, 10), (0, 5), (-10, 10), (0, 5),
             (0, 5), (0, 5), (0, 5), (0, 5),
             (0, 5), (0, 5), (0, 5), (0, 5)],
            ["rho", "lambda_supporter", "gamma_supporter",
             "lambda_opposer", "gamma_opposer",
             "c_sup_3:0", "c_sup_2:1", "c_sup_1:2", "c_sup_0:3",
             "c_opp_3:0", "c_opp_2:1", "c_opp_1:2", "c_opp_0:3"],
        ),
        "S3 v3 split + tanh saturation": (
            pred_split_saturating,
            [1.0, 2.0, 0.5, 1.0, 1.0, 0.5, 1.0],
            [(0, 10), (-10, 10), (0, 5), (0, 5),
             (-10, 10), (0, 5), (0, 5)],
            ["rho", "lambda_supporter", "gamma_supporter", "beta_supporter",
             "lambda_opposer", "gamma_opposer", "beta_opposer"],
        ),
        "S3 v3 split + extreme-reject": (
            pred_split_extreme_reject,
            [1.0, 2.0, 0.5, 1.0, 0.5, -0.5],
            [(0, 10), (-10, 10), (0, 5), (-10, 10), (0, 5), (-10, 10)],
            ["rho", "lambda_supporter", "gamma_supporter",
             "lambda_opposer", "gamma_opposer", "zeta_extreme_reject"],
        ),
        "S3 v3 full (per-coalition + saturation + extreme-reject)": (
            pred_full,
            [1.0, 2.0, 0.5, 1.0, 1.0, 0.5, 1.0, -0.5,
             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [(0, 10), (-10, 10), (0, 5), (0, 5),
             (-10, 10), (0, 5), (0, 5), (-10, 10),
             (0, 5), (0, 5), (0, 5), (0, 5),
             (0, 5), (0, 5), (0, 5), (0, 5)],
            ["rho", "lambda_supporter", "gamma_supporter", "beta_supporter",
             "lambda_opposer", "gamma_opposer", "beta_opposer",
             "zeta_extreme_reject",
             "c_sup_3:0", "c_sup_2:1", "c_sup_1:2", "c_sup_0:3",
             "c_opp_3:0", "c_opp_2:1", "c_opp_1:2", "c_opp_0:3"],
        ),
    }

    # Track best-fit predictions per model spec for residual diagnostics.
    best_preds = {}
    for name, (fn, init, bounds, par_names) in specs.items():
        res, par, q = fit(name, y, fn, init, bounds, delta_true, delta_pred)
        res["study"] = "Study 3"
        res["parameter_names"] = json.dumps(par_names)
        if per_model_id is not None:
            res["model_subset"] = per_model_id
        fits.append(res)
        params[name] = {"parameter_names": par_names, "parameters": par}
        best_preds[name] = q

    # Residual diagnostics: write residuals (delta_true - delta_pred) by
    # coalition condition and by model id for the best-fitting spec.
    best_name = min(best_preds.keys(), key=lambda k: ce(y, best_preds[k]))
    q_best = best_preds[best_name]
    delta_pred_arr = delta_pred(q_best)
    residual = delta_true - delta_pred_arr

    diag = {
        "best_model": best_name,
        "n": int(n),
        "by_coalition": {},
        "by_model": {},
    }
    for c_idx, c_name in enumerate(COALITION_LEVELS):
        m = coalition_idx == c_idx
        if m.sum() > 0:
            diag["by_coalition"][c_name] = {
                "n": int(m.sum()),
                "mean_residual": float(residual[m].mean()),
                "std_residual": float(residual[m].std()),
                "mean_abs_residual": float(np.abs(residual[m]).mean()),
            }
    for mid in np.unique(model_ids):
        m = model_ids == mid
        if m.sum() > 0:
            diag["by_model"][str(mid)] = {
                "n": int(m.sum()),
                "mean_residual": float(residual[m].mean()),
                "std_residual": float(residual[m].std()),
                "mean_abs_residual": float(np.abs(residual[m]).mean()),
            }
    params["_diagnostics"] = diag
    return fits, params


def write_csv(rows, path: Path):
    if not rows:
        return
    fields = ["study", "model_subset", "model", "n", "loss_cross_entropy", "r2_target_metric", "parameter_names", "parameters"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main():
    global OPT_N_RANDOM, OPT_N_COORD
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("/mnt/data"),
                    help="Directory containing result jsonl files. If you pass transcripts.jsonl directly, only Study 3 can run.")
    ap.add_argument("--out-dir", type=Path, default=Path("bayes_crp_fit_outputs"))
    ap.add_argument("--studies", default="1,2,3", help="Comma-separated subset, e.g. '3' or '1,2,3'.")
    ap.add_argument("--study1-opposing-only", action="store_true")
    ap.add_argument("--optimizer-random", type=int, default=450, help="Random starts for the no-SciPy optimizer. Increase to 800-1500 for more stable high-dimensional Study 3 fits.")
    ap.add_argument("--optimizer-coord", type=int, default=80, help="Coordinate-search iterations for the no-SciPy optimizer.")
    ap.add_argument("--per-model", action="store_true",
                    help="If set, fit Study 3 models separately for each LLM. With "
                         "--summaries, model identity is inferred from the filename "
                         "stem (e.g., summaries_phi.jsonl -> 'phi'). With transcripts, "
                         "the model field on each record is used.")
    ap.add_argument("--summaries", type=str, default=None,
                    help="Path to summaries-format input. Accepts: a single file, "
                         "a directory containing summaries_<model>.jsonl files, or "
                         "a glob pattern (e.g., '/path/summaries_*.jsonl'). When set, "
                         "Study 3 reads from these files instead of transcripts.jsonl. "
                         "Studies 1 and 2 are unaffected and still need --data-dir.")
    args = ap.parse_args()
    OPT_N_RANDOM = args.optimizer_random
    OPT_N_COORD = args.optimizer_coord

    # If user passes a jsonl file directly, treat its parent as data_dir and run Study 3 by default.
    file_arg = args.data_dir if args.data_dir.is_file() else None
    data_dir = args.data_dir.parent if file_arg else args.data_dir
    if file_arg and "transcripts" in file_arg.name and args.studies == "1,2,3":
        args.studies = "3"

    # Resolve --summaries into a concrete list of file paths, if provided.
    summaries_paths: Optional[List[Path]] = None
    if args.summaries is not None:
        summaries_paths = _resolve_summaries_paths(args.summaries)
        if not summaries_paths:
            raise FileNotFoundError(
                f"--summaries '{args.summaries}' did not match any files."
            )
        print(f"[summaries] Using {len(summaries_paths)} file(s):")
        for p in summaries_paths:
            print(f"  - {p}")
        # If --summaries is set without explicit --studies, default to Study 3.
        if args.studies == "1,2,3":
            args.studies = "3"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    requested = {s.strip() for s in args.studies.split(",") if s.strip()}

    rows, params = [], {}
    base = None
    if requested & {"1", "2"}:
        base_path = find_file(data_dir, ["baseline_outputs_explicit_format.jsonl", "baseline_outputs_explicit_format(1).jsonl"])
        if base_path is None:
            raise FileNotFoundError("Study 1/2 need baseline_outputs_explicit_format.jsonl in --data-dir")
        base = load_baselines(base_path)

    if "1" in requested:
        r, p = study1(data_dir, base, args.study1_opposing_only)
        rows.extend(r); params.update(p)
    if "2" in requested:
        r, p = study2(data_dir, base)
        rows.extend(r); params.update(p)
    if "3" in requested:
        # Pooled fit (all models / all summaries files).
        r, p = study3(
            file_arg if file_arg else data_dir,
            summaries_paths=summaries_paths,
        )
        for rec in r:
            rec["model_subset"] = "POOLED"
        rows.extend(r); params["POOLED"] = p

        # Per-model fits.
        if args.per_model:
            if summaries_paths is not None:
                # Model ids come from filename stems.
                model_ids_seen = set()
                for p in summaries_paths:
                    stem = p.stem
                    for prefix in ("summaries_", "transcripts_", "summary_"):
                        if stem.startswith(prefix):
                            stem = stem[len(prefix):]
                            break
                    if stem:
                        model_ids_seen.add(stem)
            else:
                # Model ids come from the transcripts file itself.
                transcripts_path = file_arg if file_arg else find_file(
                    data_dir, ["transcripts.jsonl", "transcripts(1).jsonl"]
                )
                model_ids_seen = set()
                for o in read_jsonl(transcripts_path):
                    m_id = (
                        o.get("model")
                        or o.get("model_id")
                        or o.get("model_name")
                        or (o.get("metadata") or {}).get("model")
                        or (o.get("metadata") or {}).get("model_id")
                    )
                    if m_id:
                        model_ids_seen.add(str(m_id))

            print(f"\n[per-model] Found {len(model_ids_seen)} model id(s): {sorted(model_ids_seen)}")
            for m_id in sorted(model_ids_seen):
                print(f"\n[per-model] Fitting Study 3 for model = {m_id}")
                try:
                    r_m, p_m = study3(
                        file_arg if file_arg else data_dir,
                        per_model_id=m_id,
                        summaries_paths=summaries_paths,
                    )
                    for rec in r_m:
                        rec["model_subset"] = m_id
                    rows.extend(r_m); params[m_id] = p_m
                except Exception as e:
                    print(f"[per-model] Skipping {m_id}: {e}")

    rows = sorted(rows, key=lambda r: (r.get("study", ""), r.get("model_subset", ""), r.get("loss_cross_entropy", float("inf"))))
    write_csv(rows, args.out_dir / "fit_summary.csv")
    with (args.out_dir / "fit_params.json").open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    # Pretty print without pandas.
    for r in rows:
        r2_text = "" if r.get("r2_target_metric") == "" else f", R2={r.get('r2_target_metric', ''):.3f}"
        subset = r.get("model_subset", "")
        subset_tag = f"[{subset}] " if subset else ""
        print(f"[{r['study']}] {subset_tag}{r['model']}: CE={r['loss_cross_entropy']:.4f}{r2_text}")
    print("\nWrote:", args.out_dir / "fit_summary.csv")
    print("Wrote:", args.out_dir / "fit_params.json")


def _resolve_summaries_paths(spec: str) -> List[Path]:
    """Resolve a --summaries spec into a list of jsonl file paths.

    Accepts a single file, a directory (in which case all summaries_*.jsonl
    files are picked up), or a glob pattern.
    """
    p = Path(spec)
    # Plain file.
    if p.is_file():
        return [p]
    # Directory: pick up summaries_*.jsonl by convention.
    if p.is_dir():
        candidates = sorted(p.glob("summaries_*.jsonl"))
        if not candidates:
            candidates = sorted(p.glob("*.jsonl"))
        return candidates
    # Glob pattern: match against parent directory.
    parent = p.parent if p.parent != Path("") else Path(".")
    pattern = p.name
    if "*" in pattern or "?" in pattern or "[" in pattern:
        return sorted(parent.glob(pattern))
    return []


if __name__ == "__main__":
    main()