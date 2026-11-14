"""
Motor compartido de simulación Monte Carlo dinámico.

Evoluciona los ratings Elo y una componente de forma reciente (`DFORM`, centrada
en cero respecto al corte t0) de forma cronológica partido a partido dentro de
cada trayectoria simulada, manteniendo coherencia exacta entre resultado (1X2)
y goles simulados para los desempates de liga y eliminatorias.
"""

from __future__ import annotations

import hashlib
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from comun.motor_comun import (
    corte_test_temporada,
    etiquetas_y_orden_1x2,
    hash_modulos_comunes,
)

MC_DINAMICO_VERSION = "mc_dinamico_v1"

# Fallback teórico si la calibración empírica no tiene muestra válida: ln(10)/400 ≈ 0.005756 repartido en [L, E, V]
_B_ELO_FALLBACK = np.array([0.00288, 0.0, -0.00288], dtype=np.float64)


def clave_cache_simulacion(*partes) -> str:
    """Construye una huella SHA-256 estable que incluye hash_modulos_comunes y versión del MC dinámico."""
    fp = hashlib.sha256()
    fp.update(MC_DINAMICO_VERSION.encode("utf-8"))
    fp.update(hash_modulos_comunes().encode("utf-8"))
    for parte in partes:
        if parte is None:
            continue
        fp.update(b"|")
        fp.update(str(parte).encode("utf-8"))
    return fp.hexdigest()


def clave_cache_liga(motor_file, data_dir, n_sims: int, modelo_tipo: str, fijos=None) -> str:
    """Huella completa de caché para simulacion_mc.pkl usando motor, datos, fixture y comun/."""
    fp = hashlib.sha256()
    fp.update(MC_DINAMICO_VERSION.encode("utf-8"))
    fp.update(hash_modulos_comunes().encode("utf-8"))
    try:
        fp.update(hashlib.sha256(open(motor_file, "rb").read()).hexdigest().encode("utf-8"))
    except Exception:
        pass
    from pathlib import Path
    data_p = Path(data_dir)
    for fname in ("partidos.csv", "fixture.csv", "equipos.csv"):
        fpath = data_p / fname
        if fpath.exists():
            try:
                st = fpath.stat()
                fp.update(f"|{fname}:{st.st_size}:{int(st.st_mtime)}".encode("utf-8"))
                if fname == "fixture.csv":
                    fp.update(hashlib.sha256(fpath.read_bytes()).hexdigest().encode("utf-8"))
            except Exception:
                pass
    fp.update(f"|{n_sims}|{modelo_tipo}".encode("utf-8"))
    if fijos is not None:
        fp.update(repr(sorted(fijos.items())).encode("utf-8"))
    return fp.hexdigest()


def extraer_elo_base(M: dict, eqs, default: float = 1500.0) -> np.ndarray:
    """Extrae un vector float64 con el Elo vigente al corte t0 para cada equipo en `eqs`."""
    elos_map = {}
    tracker = M.get("tracker") if isinstance(M, dict) else None
    if tracker is not None:
        raw = getattr(tracker, "elos", None)
        if not isinstance(raw, dict) or not raw:
            raw = getattr(tracker, "elo", None)
        if isinstance(raw, dict):
            elos_map = raw
    if not elos_map and isinstance(M, dict):
        if isinstance(M.get("elos"), dict):
            elos_map = M["elos"]
        elif isinstance(M.get("states"), pd.DataFrame) and "elo" in M["states"].columns:
            elos_map = M["states"]["elo"].to_dict()

    out = np.empty(len(eqs), dtype=np.float64)
    for i, eq in enumerate(eqs):
        try:
            out[i] = float(elos_map.get(eq, default))
        except Exception:
            out[i] = float(default)
    return out


def calibrar_sensibilidad_dinamica(M: dict, liga: str = None, es_binario: bool = False) -> dict:
    """Calibra (b_elo, b_form, b_gol, b_elo_bin) usando exclusivamente temporadas < test_desde (sin fuga)."""
    cache_field = "_sens_mc_bin" if es_binario else "_sens_mc_1x2"
    if isinstance(M, dict) and cache_field in M:
        return M[cache_field]

    default_res = {
        "b_elo": _B_ELO_FALLBACK.copy(),
        "b_form": np.array([0.08, 0.0, -0.08], dtype=np.float64),
        "b_gol": 0.0012,
        "b_elo_bin": float(np.log(10.0) / 400.0),
    }

    if not isinstance(M, dict):
        return default_res

    df = M.get("df_dataset") if "df_dataset" in M else M.get("df_features")
    if df is None or not isinstance(df, pd.DataFrame) or len(df) < 30 or "elo_diff" not in df.columns:
        M[cache_field] = default_res
        return default_res

    if "temporada" in df.columns:
        test_desde = corte_test_temporada(liga, df)
        df_pre = df[df["temporada"] < test_desde]
        if len(df_pre) < 30:
            corte_idx = max(30, int(len(df) * 0.7))
            df_pre = df.iloc[:corte_idx]
    else:
        df_pre = df

    pp_raw = M.get("poisson_params")
    if pp_raw is None:
        pp_raw = M.get("poiss_params")
    pp = pp_raw.to_dict() if hasattr(pp_raw, "to_dict") else (pp_raw if isinstance(pp_raw, dict) else {})
    b_gol = abs(float(pp.get("elo", pp.get("elo_diff", pp.get("b_elo_l", pp.get("d", M.get("g_d", 0.0012)))))))
    b_gol = float(np.clip(b_gol, 0.0004, 0.0025))

    if es_binario or "target_win_local" in df_pre.columns:
        if "target_win_local" in df_pre.columns:
            y_bin = df_pre["target_win_local"].to_numpy(dtype=int)
            X_b = np.ascontiguousarray(df_pre[["elo_diff"]].fillna(0.0).to_numpy(dtype=np.float64)) / 100.0
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    lr = LogisticRegression(C=5.0, max_iter=1000).fit(X_b, y_bin)
                idx_pos = list(lr.classes_).index(1) if 1 in lr.classes_ else -1
                coef = float(lr.coef_[0, 0] / 100.0) if len(lr.classes_) == 2 else float(np.log(10.0) / 400.0)
                if idx_pos == 0:
                    coef = -coef
                b_elo_bin = float(np.clip(coef, 0.002, 0.012)) if coef > 0 else float(np.log(10.0) / 400.0)
            except Exception:
                b_elo_bin = float(np.log(10.0) / 400.0)
        else:
            b_elo_bin = float(np.log(10.0) / 400.0)

        res = {
            "b_elo": _B_ELO_FALLBACK.copy(),
            "b_form": np.array([0.08, 0.0, -0.08], dtype=np.float64),
            "b_gol": b_gol,
            "b_elo_bin": b_elo_bin,
        }
        M[cache_field] = res
        return res

    # Competiciones 1X2: etiquetas_y_orden_1x2 siempre normaliza 0=Local, 1=Empate, 2=Visita
    try:
        y_pre, _ = etiquetas_y_orden_1x2(M, df_pre)
    except Exception:
        M[cache_field] = default_res
        return default_res

    use_form = "form_diff" in df_pre.columns
    cols = ["elo_diff"] + (["form_diff"] if use_form else [])
    X = np.nan_to_num(np.ascontiguousarray(df_pre[cols].to_numpy(dtype=np.float64)), nan=0.0, posinf=0.0, neginf=0.0)
    X[:, 0] /= 100.0

    b_elo = _B_ELO_FALLBACK.copy()
    b_form = np.array([0.08, 0.0, -0.08], dtype=np.float64)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lr = LogisticRegression(C=5.0, max_iter=1000).fit(X, y_pre)
        coefs = lr.coef_.copy()
        clases = list(lr.classes_)
        tmp_elo = np.zeros(3, dtype=np.float64)
        tmp_form = np.zeros(3, dtype=np.float64)
        for idx_cls, cls_val in enumerate((0, 1, 2)):
            if cls_val in clases:
                j = clases.index(cls_val)
                tmp_elo[idx_cls] = coefs[j, 0] / 100.0
                if use_form and coefs.shape[1] > 1:
                    tmp_form[idx_cls] = coefs[j, 1]
        tmp_elo -= tmp_elo.mean()
        tmp_form -= tmp_form.mean()
        if (tmp_elo[0] - tmp_elo[2]) > 0.001:
            b_elo = tmp_elo
        if use_form and (tmp_form[0] - tmp_form[2]) > 0.0:
            b_form = np.clip(tmp_form, -0.35, 0.35)
    except Exception:
        pass

    res = {
        "b_elo": b_elo,
        "b_form": b_form,
        "b_gol": b_gol,
        "b_elo_bin": float(np.log(10.0) / 400.0),
    }
    M[cache_field] = res
    return res


def _extraer_prediccion_base(val, predecir_fallback=None, local=None, visita=None):
    """Normaliza distintos formatos de `PREDS[(l, v)]` a `(p_1x2[3], la, lb)` en orden [Local, Empate, Visita]."""
    if val is None and predecir_fallback is not None:
        val = predecir_fallback(local, visita)

    if isinstance(val, np.ndarray) and val.ndim == 1 and len(val) >= 3:
        p = np.asarray(val[:3], dtype=np.float64)
        p = np.clip(p, 1e-6, 1.0 - 1e-6) if (np.isfinite(p).all() and p.sum() > 0) else np.array([0.45, 0.27, 0.28])
        p /= p.sum()
        return p, 1.35, 1.05

    if isinstance(val, np.ndarray) and val.ndim == 2:
        grid = np.asarray(val, dtype=np.float64)
        s = grid.sum()
        if s > 0:
            grid = grid / s
        p_l = float(np.tril(grid, -1).sum())
        p_e = float(np.trace(grid))
        p_v = float(np.triu(grid, 1).sum())
        p = np.clip(np.array([p_l, p_e, p_v], dtype=np.float64), 1e-6, 1.0 - 1e-6)
        p /= p.sum()
        la = float(np.sum(np.arange(grid.shape[0]) * grid.sum(axis=1)))
        lb = float(np.sum(np.arange(grid.shape[1]) * grid.sum(axis=0)))
        return p, max(la, 0.15), max(lb, 0.10)

    if isinstance(val, (tuple, list)) and len(val) >= 3:
        p = np.asarray(val[0], dtype=np.float64).ravel()[:3]
        if len(p) == 3 and np.isfinite(p).all() and p.sum() > 0:
            p = np.clip(p, 1e-6, 1.0 - 1e-6)
            p /= p.sum()
        else:
            p = np.array([0.45, 0.27, 0.28], dtype=np.float64)
        la = float(val[1]) if np.isfinite(val[1]) else 1.35
        lb = float(val[2]) if np.isfinite(val[2]) else 1.05
        return p, max(la, 0.15), max(lb, 0.10)

    return np.array([0.45, 0.27, 0.28], dtype=np.float64), 1.35, 1.05


def muestrear_goles_coherentes(rng: np.random.Generator, la, lb, w_l: np.ndarray, w_e: np.ndarray, w_v: np.ndarray):
    """Muestrea goles Poisson garantizando coherencia estricta con el resultado (w_l, w_e, w_v)."""
    n_sims = len(w_l)
    gl = rng.poisson(la, size=n_sims).astype(np.int16)
    gv = rng.poisson(lb, size=n_sims).astype(np.int16)
    la_is_arr = isinstance(la, np.ndarray) and la.ndim > 0
    lb_is_arr = isinstance(lb, np.ndarray) and lb.ndim > 0

    for _ in range(3):
        bad = (w_l & (gl <= gv)) | (w_e & (gl != gv)) | (w_v & (gl >= gv))
        if not bad.any():
            break
        nb = int(bad.sum())
        gl[bad] = rng.poisson(la[bad] if la_is_arr else la, size=nb).astype(np.int16)
        gv[bad] = rng.poisson(lb[bad] if lb_is_arr else lb, size=nb).astype(np.int16)

    m_wl = w_l & (gl <= gv)
    if m_wl.any():
        gl[m_wl] = gv[m_wl] + 1
    m_wv = w_v & (gl >= gv)
    if m_wv.any():
        gv[m_wv] = gl[m_wv] + 1
    m_we = w_e & (gl != gv)
    if m_we.any():
        g_avg = ((gl[m_we].astype(np.int32) + gv[m_we].astype(np.int32)) // 2).astype(np.int16)
        gl[m_we] = g_avg
        gv[m_we] = g_avg
    return gl, gv


def actualizar_elo_vec(
    elo_l: np.ndarray,
    elo_v: np.ndarray,
    gl,
    gv,
    k_liga: float = 35.0,
    home_adv: float = 55.0,
    mov_mode: str = "std",
) -> np.ndarray:
    """Calcula el delta Elo vectorizado para un partido dado gl vs gv y el multiplicador MOV."""
    dr = (elo_l + np.float32(home_adv)) - elo_v
    we = 1.0 / (1.0 + np.power(np.float32(10.0), -dr / np.float32(400.0)))
    sh = np.where(gl > gv, np.float32(1.0), np.where(gl == gv, np.float32(0.5), np.float32(0.0)))
    gd = np.abs(gl - gv).astype(np.float32)

    if mov_mode == "none":
        mult = np.float32(1.0)
    elif mov_mode == "arg":
        mult = (np.float32(11.0) + gd) / np.float32(8.0)
    elif mov_mode == "538":
        elo_wdiff = np.where(gl >= gv, np.abs(elo_l - elo_v), np.abs(elo_v - elo_l))
        k_eff = np.float32(20.0) * np.power(gd + np.float32(3.0), np.float32(0.8)) / (
            np.float32(7.5) + np.float32(0.006) * elo_wdiff
        )
        return (k_eff * (sh - we)).astype(np.float32)
    else:  # "std": estándar usado en ligas europeas, UCL, Chile, México y Mundial
        mult = np.where(
            gd <= np.float32(1.0),
            np.float32(1.0),
            np.where(gd == np.float32(2.0), np.float32(1.5), np.float32(1.75) + (gd - np.float32(3.0)) / np.float32(8.0)),
        )
    return (np.float32(k_liga) * mult * (sh - we)).astype(np.float32)


def desempate_y_orden(
    PTS: np.ndarray,
    GF: np.ndarray,
    GC: np.ndarray,
    PG: np.ndarray = None,
    *,
    rng: np.random.Generator,
    peso_pts: float = 1e6,
    peso_pg: float = 0.0,
    peso_dg: float = 1e3,
    peso_gf: float = 1.0,
) -> np.ndarray:
    """Ordena posiciones (descendente) usando float64 y jitter < 0.01 para evitar sesgo alfabético en empates exactos."""
    DG = GF.astype(np.float64) - GC.astype(np.float64)
    key = (
        PTS.astype(np.float64) * peso_pts
        + DG * peso_dg
        + GF.astype(np.float64) * peso_gf
        + rng.random(PTS.shape) * 0.01
    )
    if PG is not None and peso_pg:
        key = key + PG.astype(np.float64) * peso_pg
    return np.argsort(-key, axis=0)


def _ordenar_fix_cronologico(fix_df: pd.DataFrame) -> pd.DataFrame:
    if fix_df is None or len(fix_df) <= 1 or "fecha" not in fix_df.columns:
        return fix_df
    out = fix_df.copy()
    out["_fecha_sort"] = pd.to_datetime(out["fecha"], errors="coerce")
    out = out.sort_values("_fecha_sort", kind="mergesort").drop(columns=["_fecha_sort"]).reset_index(drop=True)
    return out


def _paso_dinamico_partido(
    rng: np.random.Generator,
    elo_l: np.ndarray,
    elo_v: np.ndarray,
    elo0_l,
    elo0_v,
    df_l: np.ndarray,
    df_v: np.ndarray,
    p0: np.ndarray,
    la0,
    lb0,
    b_elo: np.ndarray,
    b_form: np.ndarray,
    b_gol: float,
    home_adv: float,
    use_poisson_direct: bool,
):
    """Ajusta probabilidades y lambdas según desviación Elo/forma respecto a t0 y muestrea (w_l, w_e, w_v, gl, gv)."""
    n_sims = len(elo_l)
    d_elo = (elo_l - elo0_l) - (elo_v - elo0_v)
    d_form = df_l - df_v

    scale = np.exp(np.clip(d_elo * np.float32(b_gol), np.float32(-0.65), np.float32(0.65)))
    la_s = np.clip(la0 * scale, np.float32(0.15), np.float32(6.0))
    lb_s = np.clip(lb0 / scale, np.float32(0.10), np.float32(6.0))

    if use_poisson_direct:
        gl = rng.poisson(la_s, size=n_sims).astype(np.int16)
        gv = rng.poisson(lb_s, size=n_sims).astype(np.int16)
        w_l = gl > gv
        w_e = gl == gv
        w_v = gl < gv
        # Expectativa local aproximada para forma cuando se simula Poisson directo
        dr_tot = (elo_l + np.float32(home_adv)) - elo_v
        e_pts_l = np.float32(2.6) / (np.float32(1.0) + np.power(np.float32(10.0), -dr_tot / np.float32(400.0))) + np.float32(0.25)
        e_pts_v = np.float32(2.6) - e_pts_l
        return w_l, w_e, w_v, gl, gv, e_pts_l, e_pts_v

    log_p0 = np.log(np.clip(p0, np.float32(1e-6), np.float32(1.0 - 1e-6)))
    if log_p0.ndim == 1:
        logits = log_p0[None, :] + d_elo[:, None] * b_elo[None, :] + d_form[:, None] * b_form[None, :]
    else:
        logits = log_p0 + d_elo[:, None] * b_elo[None, :] + d_form[:, None] * b_form[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    exps = np.exp(logits)
    pk = exps / exps.sum(axis=1, keepdims=True)
    p_l, p_d = pk[:, 0], pk[:, 1]

    u = rng.random(n_sims, dtype=np.float32)
    w_l = u < p_l
    w_v = u >= (p_l + p_d)
    w_e = ~(w_l | w_v)
    gl, gv = muestrear_goles_coherentes(rng, la_s, lb_s, w_l, w_e, w_v)
    e_pts_l = np.float32(3.0) * p_l + p_d
    e_pts_v = np.float32(3.0) * pk[:, 2] + p_d
    return w_l, w_e, w_v, gl, gv, e_pts_l, e_pts_v


def simular_partido_vec_estado(
    rng: np.random.Generator,
    ELO: np.ndarray,
    DFORM: np.ndarray,
    elo0: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    PREDS_arr_fn,
    sens: dict,
    k_liga: float = 35.0,
    home_adv: float = 55.0,
    mov_mode: str = "std",
    alpha_f: float = 0.20,
):
    """Simula un partido entre pares dinámicos (idx_a[s], idx_b[s]) por trayectoria y actualiza ELO y DFORM in-place."""
    n_sims = len(idx_a)
    sims = np.arange(n_sims)
    p0_all, la0_all, lb0_all = PREDS_arr_fn(idx_a, idx_b)

    elo_l = ELO[idx_a, sims]
    elo_v = ELO[idx_b, sims]
    df_l = DFORM[idx_a, sims]
    df_v = DFORM[idx_b, sims]
    b_elo = np.asarray(sens["b_elo"], dtype=np.float32)
    b_form = np.asarray(sens["b_form"], dtype=np.float32)
    b_gol = float(sens["b_gol"])

    w_l, w_e, w_v, gl, gv, e_pts_l, e_pts_v = _paso_dinamico_partido(
        rng=rng,
        elo_l=elo_l,
        elo_v=elo_v,
        elo0_l=elo0[idx_a].astype(np.float32),
        elo0_v=elo0[idx_b].astype(np.float32),
        df_l=df_l,
        df_v=df_v,
        p0=p0_all.astype(np.float32),
        la0=la0_all.astype(np.float32),
        lb0=lb0_all.astype(np.float32),
        b_elo=b_elo,
        b_form=b_form,
        b_gol=b_gol,
        home_adv=home_adv,
        use_poisson_direct=False,
    )

    delta = actualizar_elo_vec(elo_l, elo_v, gl, gv, k_liga=k_liga, home_adv=home_adv, mov_mode=mov_mode)
    np.add.at(ELO, (idx_a, sims), delta)
    np.add.at(ELO, (idx_b, sims), -delta)

    pts_l = np.where(w_l, np.float32(3.0), np.where(w_e, np.float32(1.0), np.float32(0.0)))
    pts_v = np.where(w_v, np.float32(3.0), np.where(w_e, np.float32(1.0), np.float32(0.0)))
    af = np.float32(alpha_f)
    DFORM[idx_a, sims] = (np.float32(1.0) - af) * df_l + af * (pts_l - e_pts_l)
    DFORM[idx_b, sims] = (np.float32(1.0) - af) * df_v + af * (pts_v - e_pts_v)
    return gl, gv, w_l, w_e, w_v


def simular_fixture_dinamico(
    M: dict,
    fix_df: pd.DataFrame,
    eqs: list[str],
    init_pts: np.ndarray,
    init_gf: np.ndarray,
    init_gc: np.ndarray,
    PREDS: dict,
    *,
    fijos: dict = None,
    n_sims: int = 3000,
    k_liga: float = 35.0,
    home_adv: float = 55.0,
    mov_mode: str = "std",
    init_pg: np.ndarray = None,
    peso_pts: float = 1e6,
    peso_pg: float = 0.0,
    peso_dg: float = 1e3,
    peso_gf: float = 1.0,
    use_poisson_direct: bool = False,
    predecir_fallback=None,
    liga: str = None,
    seed: int = None,
    alpha_f: float = 0.20,
    track_terminal_states: bool = False,
) -> dict:
    """Simula cronológicamente un fixture completo evolucionando Elo y forma (DFORM) dentro de cada simulación."""
    rng = np.random.default_rng(seed)
    n_eq = len(eqs)
    ix = {e: i for i, e in enumerate(eqs)}

    PTS = np.tile(np.asarray(init_pts, dtype=np.float32)[:, None], (1, n_sims))
    GF = np.tile(np.asarray(init_gf, dtype=np.float32)[:, None], (1, n_sims))
    GC = np.tile(np.asarray(init_gc, dtype=np.float32)[:, None], (1, n_sims))
    PG = np.tile(np.asarray(init_pg, dtype=np.float32)[:, None], (1, n_sims)) if init_pg is not None else None

    elo0 = extraer_elo_base(M, eqs).astype(np.float32)
    ELO = np.tile(elo0[:, None], (1, n_sims))
    DFORM = np.zeros((n_eq, n_sims), dtype=np.float32)

    sens = calibrar_sensibilidad_dinamica(M, liga=liga, es_binario=False)
    b_elo = np.asarray(sens["b_elo"], dtype=np.float32)
    b_form = np.asarray(sens["b_form"], dtype=np.float32)
    b_gol = float(sens["b_gol"])
    af = np.float32(alpha_f)

    fix_ord = _ordenar_fix_cronologico(fix_df)
    if fix_ord is not None and not fix_ord.empty:
        for r in fix_ord.itertuples(index=False):
            local, visita = r.local, r.visita
            if local not in ix or visita not in ix:
                continue
            li, vi = ix[local], ix[visita]

            if fijos and (local, visita) in fijos:
                fg_l, fg_v = fijos[(local, visita)]
                gl = np.full(n_sims, int(fg_l), dtype=np.int16)
                gv = np.full(n_sims, int(fg_v), dtype=np.int16)
                w_l = gl > gv
                w_e = gl == gv
                w_v = gl < gv
                p0, _, _ = _extraer_prediccion_base(PREDS.get((local, visita)), predecir_fallback, local, visita)
                e_pts_l = np.float32(3.0 * p0[0] + p0[1])
                e_pts_v = np.float32(3.0 * p0[2] + p0[1])
            else:
                p0, la0, lb0 = _extraer_prediccion_base(PREDS.get((local, visita)), predecir_fallback, local, visita)
                w_l, w_e, w_v, gl, gv, e_pts_l, e_pts_v = _paso_dinamico_partido(
                    rng=rng,
                    elo_l=ELO[li],
                    elo_v=ELO[vi],
                    elo0_l=elo0[li],
                    elo0_v=elo0[vi],
                    df_l=DFORM[li],
                    df_v=DFORM[vi],
                    p0=p0.astype(np.float32),
                    la0=np.float32(la0),
                    lb0=np.float32(lb0),
                    b_elo=b_elo,
                    b_form=b_form,
                    b_gol=b_gol,
                    home_adv=home_adv,
                    use_poisson_direct=use_poisson_direct,
                )

            pts_l = np.where(w_l, np.float32(3.0), np.where(w_e, np.float32(1.0), np.float32(0.0)))
            pts_v = np.where(w_v, np.float32(3.0), np.where(w_e, np.float32(1.0), np.float32(0.0)))
            PTS[li] += pts_l
            PTS[vi] += pts_v
            GF[li] += gl
            GF[vi] += gv
            GC[li] += gv
            GC[vi] += gl
            if PG is not None:
                PG[li] += w_l
                PG[vi] += w_v

            delta = actualizar_elo_vec(ELO[li], ELO[vi], gl, gv, k_liga=k_liga, home_adv=home_adv, mov_mode=mov_mode)
            ELO[li] += delta
            ELO[vi] -= delta
            DFORM[li] = (np.float32(1.0) - af) * DFORM[li] + af * (pts_l - e_pts_l)
            DFORM[vi] = (np.float32(1.0) - af) * DFORM[vi] + af * (pts_v - e_pts_v)

    order = desempate_y_orden(
        PTS, GF, GC, PG=PG, rng=rng,
        peso_pts=peso_pts, peso_pg=peso_pg, peso_dg=peso_dg, peso_gf=peso_gf,
    )
    out = {
        "eqs": eqs,
        "ix": ix,
        "order": order,
        "PTS": PTS,
        "GF": GF,
        "GC": GC,
        "DG": GF - GC,
        "PG": PG,
    }
    if track_terminal_states:
        out.update({"ELO": ELO, "DFORM": DFORM, "elo0": elo0, "sens": sens})
    return out


def simular_fixture_desde_tabla(
    M: dict,
    fix_df: pd.DataFrame,
    tab: pd.DataFrame,
    PREDS: dict,
    *,
    fijos: dict = None,
    n_sims: int = 3000,
    k_liga: float = 35.0,
    home_adv: float = 55.0,
    mov_mode: str = "std",
    col_pts: str = "puntos",
    col_dg: str = "dif_goles",
    col_gf: str = "goles_favor",
    col_gc: str = None,
    col_pg: str = None,
    peso_pts: float = 1e6,
    peso_pg: float = 0.0,
    peso_dg: float = 1e3,
    peso_gf: float = 1.0,
    usar_lambdas_elo: bool = False,
    use_poisson_direct: bool = False,
    predecir_fallback=None,
    liga: str = None,
    seed: int = None,
    track_terminal_states: bool = False,
) -> dict:
    """Adapta una tabla actual (`obtener_tabla_actual`) y un fixture a `simular_fixture_dinamico`."""
    tab_eqs = list(tab["equipo"]) if (tab is not None and not tab.empty) else []
    seen = set(tab_eqs)
    eqs = list(tab_eqs)
    if fix_df is not None and not fix_df.empty:
        for e in sorted(set(fix_df["local"]).union(set(fix_df["visita"]))):
            if e not in seen:
                eqs.append(e)
                seen.add(e)

    n_eq = len(eqs)
    init_pts = np.zeros(n_eq, dtype=np.float32)
    init_gf = np.zeros(n_eq, dtype=np.float32)
    init_gc = np.zeros(n_eq, dtype=np.float32)
    init_pg = np.zeros(n_eq, dtype=np.float32) if col_pg is not None else None

    if tab is not None and not tab.empty:
        row_map = tab.set_index("equipo").to_dict("index")
        for i, e in enumerate(eqs):
            r = row_map.get(e)
            if r is None:
                continue
            pts_val = float(r.get(col_pts, 0.0))
            gf_val = float(r.get(col_gf, 0.0))
            if col_gc and col_gc in r:
                gc_val = float(r[col_gc])
            else:
                gc_val = gf_val - float(r.get(col_dg, 0.0))
            init_pts[i] = pts_val
            init_gf[i] = gf_val
            init_gc[i] = gc_val
            if init_pg is not None:
                init_pg[i] = float(r.get(col_pg, 0.0))

    preds_use = PREDS
    if usar_lambdas_elo and PREDS is not None and fix_df is not None and not fix_df.empty:
        elos = getattr(M.get("tracker"), "elos", {}) if isinstance(M, dict) else {}
        g_c = float(M.get("g_const", 0.1))
        g_d = float(M.get("g_d", 0.0012))
        g_h = float(M.get("g_home", 0.25))
        preds_use = dict(PREDS)
        for r in fix_df.itertuples(index=False):
            pair = (r.local, r.visita)
            val = preds_use.get(pair)
            if val is None and predecir_fallback is not None:
                val = predecir_fallback(r.local, r.visita)
            if isinstance(val, np.ndarray) and val.ndim == 1 and len(val) >= 3:
                ed = float(elos.get(r.local, 1500.0)) - float(elos.get(r.visita, 1500.0))
                la = float(np.clip(np.exp(g_c + g_d * ed + g_h), 0.15, 5.5))
                lb = float(np.clip(np.exp(g_c - g_d * ed), 0.10, 5.5))
                preds_use[pair] = (val, la, lb)

    return simular_fixture_dinamico(
        M, fix_df, eqs, init_pts, init_gf, init_gc, preds_use,
        fijos=fijos, n_sims=n_sims, k_liga=k_liga, home_adv=home_adv, mov_mode=mov_mode,
        init_pg=init_pg, peso_pts=peso_pts, peso_pg=peso_pg, peso_dg=peso_dg, peso_gf=peso_gf,
        use_poisson_direct=use_poisson_direct, predecir_fallback=predecir_fallback,
        liga=liga, seed=seed, track_terminal_states=track_terminal_states,
    )


def estado_inicial_desde_partidos(p_actuales: pd.DataFrame, eqs: list[str]):
    """Construye vectores (pts, gf, gc, pg) acumulados desde un DataFrame de partidos ya jugados."""
    n_eq = len(eqs)
    ix = {e: i for i, e in enumerate(eqs)}
    pts = np.zeros(n_eq, dtype=np.float32)
    gf = np.zeros(n_eq, dtype=np.float32)
    gc = np.zeros(n_eq, dtype=np.float32)
    pg = np.zeros(n_eq, dtype=np.float32)
    if p_actuales is None or p_actuales.empty:
        return pts, gf, gc, pg

    for r in p_actuales.itertuples(index=False):
        l, v = r.local, r.visita
        if l not in ix or v not in ix:
            continue
        li, vi = ix[l], ix[v]
        gl, gv = float(r.goles_local), float(r.goles_visita)
        if not (np.isfinite(gl) and np.isfinite(gv)):
            continue
        gf[li] += gl
        gc[li] += gv
        gf[vi] += gv
        gc[vi] += gl
        if gl > gv:
            pts[li] += 3.0
            pg[li] += 1.0
        elif gl == gv:
            pts[li] += 1.0
            pts[vi] += 1.0
        else:
            pts[vi] += 3.0
            pg[vi] += 1.0
    return pts, gf, gc, pg


def paso_poisson_dinamico_vec(
    rng: np.random.Generator,
    LA: np.ndarray,
    LB: np.ndarray,
    ELO: np.ndarray,
    elo0: np.ndarray,
    idx_l: np.ndarray,
    idx_v: np.ndarray,
    b_gol: float,
    k_liga: float = 35.0,
    home_adv: float = 60.0,
    mov_mode: str = "std",
):
    """Simula un partido con pares dinámicos (idx_l[s], idx_v[s]) vía Poisson ajustado por delta Elo y actualiza ELO in-place."""
    sims = np.arange(len(idx_l))
    elo_l = ELO[idx_l, sims]
    elo_v = ELO[idx_v, sims]
    d_elo = (elo_l - elo0[idx_l]) - (elo_v - elo0[idx_v])
    scale = np.exp(np.clip(d_elo * np.float32(b_gol), np.float32(-0.65), np.float32(0.65)))
    la = np.clip(LA[idx_l, idx_v].astype(np.float32) * scale, np.float32(0.15), np.float32(6.0))
    lb = np.clip(LB[idx_l, idx_v].astype(np.float32) / scale, np.float32(0.10), np.float32(6.0))
    gl = rng.poisson(la).astype(np.int16)
    gv = rng.poisson(lb).astype(np.int16)
    delta = actualizar_elo_vec(elo_l, elo_v, gl, gv, k_liga=k_liga, home_adv=home_adv, mov_mode=mov_mode)
    np.add.at(ELO, (idx_l, sims), delta)
    np.add.at(ELO, (idx_v, sims), -delta)
    return gl, gv


def paso_binario_dinamico_vec(
    rng: np.random.Generator,
    ELO: np.ndarray,
    DFORM: np.ndarray,
    elo0: np.ndarray,
    idx_l: np.ndarray,
    idx_v: np.ndarray,
    sims_col: np.ndarray,
    P0: np.ndarray,
    b_elo_bin: float,
    b_form_bin: float = 0.25,
    home_adv: float = 100.0,
    sigma_margin: float = 12.5,
    mov_mode: str = "538",
    alpha_f: float = 0.15,
    fixed_win_l=None,
) -> np.ndarray:
    """Paso dinámico vectorizado para competiciones binarias (ej. NBA), actualizando ELO (MOV 538) y DFORM in-place."""
    n = len(idx_l)
    elo_l = ELO[idx_l, sims_col]
    elo_v = ELO[idx_v, sims_col]
    df_l = DFORM[idx_l, sims_col]
    df_v = DFORM[idx_v, sims_col]

    d_elo = (elo_l - elo0[idx_l]) - (elo_v - elo0[idx_v])
    d_form = df_l - df_v
    p0 = np.clip(P0[idx_l, idx_v], np.float32(1e-4), np.float32(1.0 - 1e-4))
    logit = np.log(p0 / (np.float32(1.0) - p0)) + d_elo * np.float32(b_elo_bin) + d_form * np.float32(b_form_bin)
    p_dyn = np.float32(1.0) / (np.float32(1.0) + np.exp(-np.clip(logit, np.float32(-10.0), np.float32(10.0))))

    if fixed_win_l is None:
        win_l = rng.random(n, dtype=np.float32) < p_dyn
    else:
        win_l = np.full(n, bool(fixed_win_l), dtype=bool)

    # Margen positivo de victoria coherente para el multiplicador MOV FiveThirtyEight
    mov_raw = np.abs(rng.normal(loc=0.0, scale=float(sigma_margin), size=n)).astype(np.float32)
    mov_edge = np.clip(np.where(win_l, d_elo, -d_elo) * np.float32(0.02), np.float32(0.0), np.float32(4.0))
    mov = np.maximum(np.float32(1.0), mov_raw + mov_edge)
    gl = np.where(win_l, mov, np.float32(0.0))
    gv = np.where(win_l, np.float32(0.0), mov)

    delta = actualizar_elo_vec(elo_l, elo_v, gl, gv, k_liga=20.0, home_adv=home_adv, mov_mode=mov_mode)
    # Cada elemento k corresponde a una trayectoria sims_col[k] distinta -> indexación directa sin colisiones
    ELO[idx_l, sims_col] = elo_l + delta
    ELO[idx_v, sims_col] = elo_v - delta

    res_l = np.where(win_l, np.float32(1.0), np.float32(0.0))
    res_v = np.float32(1.0) - res_l
    exp_v = np.float32(1.0) - p_dyn
    af = np.float32(alpha_f)
    DFORM[idx_l, sims_col] = (np.float32(1.0) - af) * df_l + af * (res_l - p_dyn)
    DFORM[idx_v, sims_col] = (np.float32(1.0) - af) * df_v + af * (res_v - exp_v)
    return win_l


def simular_serie_best_of_n_vec(
    rng: np.random.Generator,
    ELO: np.ndarray,
    DFORM: np.ndarray,
    elo0: np.ndarray,
    t_home: np.ndarray,
    t_away: np.ndarray,
    P0: np.ndarray,
    b_elo_bin: float,
    b_form_bin: float = 0.25,
    home_adv: float = 100.0,
    sigma_margin: float = 12.5,
    best_of: int = 7,
    patron_local=(True, True, False, False, True, False, True),
    mov_mode: str = "538",
) -> np.ndarray:
    """Simula una serie al mejor de N partidos (ej. 2-2-1-1-1) con Elo y forma evolucionando juego a juego."""
    n_sims = len(t_home)
    need = best_of // 2 + 1
    w_h = np.zeros(n_sims, dtype=np.int8)
    w_a = np.zeros(n_sims, dtype=np.int8)

    for g in range(best_of):
        active = (w_h < need) & (w_a < need)
        if not active.any():
            break
        act_idx = np.flatnonzero(active)
        is_h_home = bool(patron_local[g])
        ia = t_home[act_idx] if is_h_home else t_away[act_idx]
        ib = t_away[act_idx] if is_h_home else t_home[act_idx]
        win_l = paso_binario_dinamico_vec(
            rng, ELO, DFORM, elo0, ia, ib, act_idx, P0,
            b_elo_bin=b_elo_bin, b_form_bin=b_form_bin,
            home_adv=home_adv, sigma_margin=sigma_margin, mov_mode=mov_mode,
        )
        h_won = win_l if is_h_home else (~win_l)
        w_h[act_idx[h_won]] += 1
        w_a[act_idx[~h_won]] += 1

    return np.where(w_h >= need, t_home, t_away)



