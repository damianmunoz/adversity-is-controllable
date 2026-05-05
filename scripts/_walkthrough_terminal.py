"""
scripts/_walkthrough_terminal.py — paseo guiado e interactivo del pipeline.

Para cada tick imprime, paso por paso (con ENTER entre cada paso), el rol
de cada bloque del diagrama:

  [Bloque 1]  Fila del parquet (lo que mide el sistema)
  [Bloque 2]  Estandarización de la observación z
  [Bloque 3a] Filtro de Kalman — paso PREDICT
  [Bloque 3b] Filtro de Kalman — paso UPDATE (innovación, ganancia,
              estado posterior)
  [Bloque 4]  Lookup del bucket de presión
  [Bloque 5a] Pesos congelados del bucket (la política aprendida)
  [Bloque 5b] Muestreo de una acción
  [Bloque 6]  Simulación del fill
  [Bloque 7]  Cálculo de la pérdida
  [Bloque 8]  Actualización (hipotética) de Hedge

Las matrices se imprimen con paréntesis Unicode (⎡ ⎤ ⎣ ⎦) y se enseña
paso a paso cómo se calcula cada producto matricial.

Uso:
    PYTHONPATH=. python scripts/_walkthrough_terminal.py
    PYTHONPATH=. python scripts/_walkthrough_terminal.py --session 7 --start-tick 800 --max-ticks 3
    PYTHONPATH=. python scripts/_walkthrough_terminal.py --auto       (sin pausas)

Sesiones disponibles: 1, 2, 3, 6, 7. (4 y 5 son fragmentos de ~70 ticks
producidos por el self-heal y se omiten — el script los detecta y avisa.)
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logging.disable(logging.CRITICAL)

from src.execution.loss import compute_loss
from src.execution.simulator import simulate_fill
from src.policy.actions import Action
from src.state.kalman_filter import KalmanConfig, KalmanFilter
from src.utils.config import load_config
from src.utils.io import iter_parquet_dir


SESSION_GAP_MS = 5_000
PRESSURE_EDGES = [-0.5, -0.2, 0.0, 0.2, 0.5]
ETA            = 0.10
LAMBDA         = 0.10


# ---------------------------------------------------------------------------
# Colores ANSI
# ---------------------------------------------------------------------------

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    UNDER   = "\033[4m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"

# permitir desactivar colores (si la terminal no los soporta)
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    for attr in dir(C):
        if not attr.startswith("_") and isinstance(getattr(C, attr), str):
            setattr(C, attr, "")


# ---------------------------------------------------------------------------
# Helpers de impresión
# ---------------------------------------------------------------------------

AUTOPLAY = False


def wait(prompt: str = "  [Presiona ENTER para continuar...]") -> None:
    if AUTOPLAY:
        print(C.DIM + prompt + " (auto)" + C.RESET)
        return
    try:
        input(C.DIM + prompt + C.RESET)
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(0)


def hr(char: str = "═", color: str = C.YELLOW) -> None:
    print(color + char * 78 + C.RESET)


def block_header(num: str, title: str) -> None:
    print()
    hr("═", C.YELLOW)
    print(C.YELLOW + C.BOLD + f"  {num}  {title}" + C.RESET)
    hr("═", C.YELLOW)


def section(title: str) -> None:
    print()
    print(C.CYAN + "─── " + title + " " + "─" * (74 - len(title)) + C.RESET)


def info(text: str) -> None:
    print(C.WHITE + text + C.RESET)


def note(text: str) -> None:
    print(C.DIM + text + C.RESET)


def good(text: str) -> None:
    print(C.GREEN + text + C.RESET)


def bad(text: str) -> None:
    print(C.RED + text + C.RESET)


def highlight(text: str) -> None:
    print(C.MAGENTA + C.BOLD + text + C.RESET)


def fmt_matrix(M, prec: int = 4, color: str = C.BLUE) -> List[str]:
    """Devuelve una lista de líneas con la matriz formateada."""
    M = np.atleast_2d(np.asarray(M, dtype=float))
    rows, cols = M.shape
    cells = [[f"{x:+.{prec}f}" for x in row] for row in M]
    widths = [max(len(cells[r][c]) for r in range(rows)) for c in range(cols)]
    body_lines = []
    for r in range(rows):
        line = "  ".join(cells[r][c].rjust(widths[c]) for c in range(cols))
        body_lines.append(line)
    if rows == 1:
        return [color + f"[ {body_lines[0]} ]" + C.RESET]
    out = [color + f"⎡ {body_lines[0]} ⎤" + C.RESET]
    for r in range(1, rows - 1):
        out.append(color + f"⎢ {body_lines[r]} ⎥" + C.RESET)
    out.append(color + f"⎣ {body_lines[-1]} ⎦" + C.RESET)
    return out


def print_matrix(M, label: str = "", prec: int = 4,
                 color: str = C.BLUE, indent: int = 4) -> None:
    lines = fmt_matrix(M, prec=prec, color=color)
    if label:
        prefix = " " * indent + C.BOLD + label + " = " + C.RESET
        rest = " " * (indent + len(label) + 3)
        print(prefix + lines[0])
        for line in lines[1:]:
            print(rest + line)
    else:
        for line in lines:
            print(" " * indent + line)


def print_eq(parts: List, prec: int = 4, color: str = C.BLUE) -> None:
    """Imprime una ecuación matricial: A op B op C = R, todo lado a lado.

    parts es una lista alternando: matriz, "op", matriz, "op", ..., "=", matriz.
    """
    blocks = []
    for p in parts:
        if isinstance(p, str):
            blocks.append([p])
        else:
            blocks.append(fmt_matrix(p, prec=prec, color=color))
    # alinear verticalmente al centro
    max_h = max(len(b) for b in blocks)
    aligned = []
    for b in blocks:
        if len(b) == max_h:
            aligned.append(b); continue
        # los strings sin colores tienen el ancho real
        # los formateados con color están "mentidos" — usamos longitud visible
        is_op = (len(b) == 1 and len(b[0]) <= 5)
        plain = b[0] if is_op else "X" * 10
        pad_top = (max_h - len(b)) // 2
        pad_bot = max_h - len(b) - pad_top
        empty = " " * len(_strip_ansi(b[0]))
        aligned.append([empty] * pad_top + b + [empty] * pad_bot)
    # imprimir línea por línea
    for row_idx in range(max_h):
        parts_row = [block[row_idx] for block in aligned]
        line = "    " + "  ".join(parts_row)
        print(line)


def _strip_ansi(s: str) -> str:
    """Quitar códigos ANSI para medir ancho visible."""
    import re
    return re.sub(r"\033\[[0-9;]*m", "", s)


def kv(label: str, value: str, color_label: str = C.WHITE,
       color_value: str = C.YELLOW, indent: int = 4) -> None:
    print(" " * indent
          + color_label + f"{label:<22} " + C.RESET
          + color_value + str(value) + C.RESET)


def print_matrix_intro(name: str, description: str, M,
                       annotations: Optional[List] = None,
                       prec: int = 4, color: str = C.BLUE) -> None:
    """Print a matrix with: name, what-it-does description, the matrix,
    and per-cell meaning annotations.

    annotations: list of (row, col, meaning_text). Use col=None for vectors.
    """
    print()
    print(C.BOLD + C.WHITE + f"   ◆ {name}  " + C.RESET
          + C.DIM + f"— {description}" + C.RESET)
    print()
    print_matrix(M, name, prec=prec, color=color, indent=6)
    if annotations:
        print()
        Mv = np.atleast_2d(np.asarray(M, dtype=float))
        for ann in annotations:
            r, c, meaning = ann
            if c is None:
                idx_str = f"{name}[{r}]"
                val = float(np.asarray(M).flatten()[r])
            else:
                idx_str = f"{name}[{r},{c}]"
                val = float(Mv[r, c])
            print(f"      {C.BOLD}{C.MAGENTA}{idx_str:<10}{C.RESET} = "
                  f"{C.YELLOW}{val:+.4f}{C.RESET}  →  {meaning}")


# ---------------------------------------------------------------------------
# Detección de sesiones
# ---------------------------------------------------------------------------

def detect_sessions(rows):
    out, start = [], 0
    for i in range(1, len(rows)):
        tg = rows[i]["ts_ms"] - rows[i - 1]["ts_ms"]
        if rows[i].get("sequence_gap") or tg > SESSION_GAP_MS:
            out.append((start, i)); start = i
    out.append((start, len(rows)))
    return out


# ---------------------------------------------------------------------------
# Bucket
# ---------------------------------------------------------------------------

def which_bucket(pressure: float) -> int:
    return bisect.bisect_right(PRESSURE_EDGES, pressure)


def bucket_range_str(b: int) -> str:
    if b == 0:
        return "(-∞, -0.500]"
    if b == len(PRESSURE_EDGES):
        return "(+0.500, +∞)"
    return f"({PRESSURE_EDGES[b-1]:+.3f}, {PRESSURE_EDGES[b]:+.3f}]"


def bucket_meaning(b: int) -> str:
    return {
        0: "presión vendedora MUY fuerte",
        1: "presión vendedora moderada",
        2: "ligera tendencia vendedora",
        3: "ligera tendencia compradora",
        4: "presión compradora moderada",
        5: "presión compradora MUY fuerte",
    }.get(b, "desconocido")


# ---------------------------------------------------------------------------
# Pasos del pipeline (cada uno espera ENTER al final si quieres pausar)
# ---------------------------------------------------------------------------

def step_block_1_parquet(curr: dict, nxt: dict, tick_idx: int) -> None:
    block_header("[BLOQUE 1]", "FILA DEL PARQUET — lo que mide el sistema")
    info("Cada segundo el pipeline produce una fila como ésta a partir del libro de\n"
         "órdenes de Binance. Esto es lo que el filtro de Kalman recibirá como entrada.")
    print()
    section("Tick actual")
    kv("ts_ms",            f"{curr['ts_ms']}")
    kv("best_bid_px",      f"${curr['best_bid_px']:,.2f}")
    kv("best_ask_px",      f"${curr['best_ask_px']:,.2f}")
    kv("spread_abs",       f"${curr['spread_abs']:.4f}")
    kv("mid_price",        f"${curr['mid_price']:,.4f}")
    kv("microprice",       f"${curr['microprice']:,.4f}")
    kv("depth_imbalance",  f"{curr['depth_imbalance']:+.4f}",
       color_value=C.GREEN if curr['depth_imbalance'] > 0 else C.RED)
    kv("ofi_l1",           f"{curr['ofi_l1']:+.4f}",
       color_value=C.GREEN if curr['ofi_l1'] > 0 else C.RED)
    kv("vol_30s",          f"{curr['vol_30s']:.6e}")
    print()
    section("Tick siguiente (sólo se usa para simular el fill)")
    kv("next mid_price",   f"${nxt['mid_price']:,.4f}")
    delta = nxt['mid_price'] - curr['mid_price']
    if delta > 0:
        kv("Δ mid",        f"+${delta:.4f}  (precio SUBIÓ)", color_value=C.GREEN)
    elif delta < 0:
        kv("Δ mid",        f"-${abs(delta):.4f}  (precio BAJÓ)", color_value=C.RED)
    else:
        kv("Δ mid",        f"$0  (sin cambio)", color_value=C.WHITE)
    print()
    note("→ Estos 3 features (depth_imbalance, ofi_l1, vol_30s) son los que entran\n"
         "  al filtro de Kalman. Los demás los usa el simulador para calcular el fill.")
    wait()


def step_block_2_standardize(curr: dict, kalman_cfg: KalmanConfig
                             ) -> np.ndarray:
    block_header("[BLOQUE 2]", "ESTANDARIZACIÓN DE LA OBSERVACIÓN")
    info("vol_30s vive en una escala diminuta (~10⁻⁵) comparada con depth_imbalance\n"
         "y ofi_l1. Si la pasamos cruda al filtro, su señal se 'pierde'. La\n"
         "transformamos a media 0 y desviación 1 antes del Kalman.")
    print()

    z_raw = np.array([curr["depth_imbalance"], curr["ofi_l1"], curr["vol_30s"]],
                     dtype=float)
    center = np.array(kalman_cfg.obs_center, dtype=float)
    scale  = np.array(kalman_cfg.obs_scale,  dtype=float)

    print_matrix_intro(
        "z_raw",
        "vector observación crudo (3×1). Las tres features que el feature pipeline\n"
        "       calculó del libro de órdenes en este tick.",
        z_raw.reshape(3, 1),
        prec=6,
        annotations=[
            (0, None, "depth_imbalance crudo (entre -1 y +1)"),
            (1, None, "ofi_l1 crudo (típicamente entre -30 y +30)"),
            (2, None, "vol_30s crudo (orden de magnitud 10⁻⁵ — diminuto)"),
        ],
    )

    print_matrix_intro(
        "center",
        "vector de centrado. Se resta a z_raw para que cada feature tenga media 0.",
        center.reshape(3, 1),
        prec=6,
        annotations=[
            (0, None, "0 → depth_imbalance ya está centrado en 0 por construcción"),
            (1, None, "0 → ofi_l1 ya tiene media ≈ 0 empíricamente"),
            (2, None, "media empírica de vol_30s — la restamos para centrar"),
        ],
    )
    print_matrix_intro(
        "scale",
        "vector de escala. Multiplica al resultado para que cada feature tenga\n"
        "       desviación estándar 1.",
        scale.reshape(3, 1),
        prec=2,
        annotations=[
            (0, None, "1 → no escalamos depth_imbalance"),
            (1, None, "1 → no escalamos ofi_l1"),
            (2, None, "1/std(vol_30s) ≈ 36876 — escalamos para que vol_30s tenga std=1"),
        ],
    )
    print()

    z_norm = (z_raw - center) * scale
    section("Aritmética por componente: z_norm = (z_raw − center) × scale")
    actions = list(kalman_cfg.obs_features)
    for i, name in enumerate(actions):
        info(f"    z_norm[{i}]  =  ({z_raw[i]:.6e} − {center[i]:.6e}) × {scale[i]:.4f}")
        info(f"               =  {z_norm[i]:+.6f}")

    print()
    section("Vector observación ESTANDARIZADO (entra al filtro)")
    print_matrix(z_norm.reshape(3, 1), "z_norm", prec=6)
    print()
    note(f"→ Nota: vol_30s pasó de {z_raw[2]:.2e} (casi cero) a {z_norm[2]:+.4f}\n"
         "  (escala razonable). Ahora el Kalman puede usarla.")
    wait()
    return z_norm


def step_block_3a_predict(x_prior, P_prior, F, Q):
    block_header("[BLOQUE 3a]", "FILTRO DE KALMAN — PASO PREDICCIÓN")
    info("'Antes de mirar la observación nueva, ¿qué creo que vale el estado oculto\n"
         "ahora, basándome en lo que veía hace un segundo?'")
    print()
    note("(Esta es la parte donde el Kalman le dice a Hedge: 'basándome en lo que vi\n"
         "antes, creo que la presión y el régimen están más o menos por aquí'.)")

    section("Entradas al paso de predicción — qué representa cada matriz")

    # ---- x_prior ----
    print_matrix_intro(
        "x_prior",
        "vector de estado oculto del tick anterior — lo que el filtro creía que\n"
        "       valían 'presión' y 'régimen' después de procesar el tick previo.",
        x_prior.reshape(2, 1),
        annotations=[
            (0, None, "presión estimada en el tick anterior"),
            (1, None, "régimen estimado en el tick anterior"),
        ],
    )

    # ---- P_prior ----
    print_matrix_intro(
        "P_prior",
        "covarianza posterior del tick anterior — cuánta incertidumbre tiene el\n"
        "       filtro sobre los valores de x_prior.",
        P_prior,
        annotations=[
            (0, 0, "varianza de la estimación de presión (más alto = más inseguro)"),
            (1, 1, "varianza de la estimación de régimen"),
            (0, 1, "covarianza presión↔régimen (cero ⇒ se estiman como independientes)"),
        ],
    )

    # ---- F ----
    print_matrix_intro(
        "F",
        "matriz de transición. Cómo evoluciona el estado oculto entre dos ticks,\n"
        "       ANTES de ver la observación nueva. Esencialmente: 'cuánto recuerdo del pasado'.",
        F,
        prec=2,
        annotations=[
            (0, 0, "presión × 0.90 → decae 10% por tick (vida media ~10 ticks)"),
            (1, 1, "régimen × 0.95 → decae sólo 5% por tick (más persistente)"),
            (0, 1, "0 → la presión NO se mezcla con el régimen en la dinámica"),
            (1, 0, "0 → el régimen NO afecta la dinámica de la presión"),
        ],
    )

    # ---- Q ----
    print_matrix_intro(
        "Q",
        "covarianza del ruido de proceso. Cuánto puede 'patearse' el estado por\n"
        "       eventos del mundo no modelados, entre dos ticks consecutivos.",
        Q,
        annotations=[
            (0, 0, "0.010 → varianza del 'pateo' aleatorio en presión por tick"),
            (1, 1, "0.005 → 'pateo' en régimen — la mitad, porque el régimen es más estable"),
            (0, 1, "0 → los pateos en presión y régimen no están correlacionados"),
        ],
    )
    wait()

    print()
    section("Cálculo 1: x_pred = F · x_prior")
    info("   F es diagonal — multiplicar es directo, componente por componente:")
    info(f"     x_pred[0]  =  {F[0,0]:.2f} × {x_prior[0]:+.4f}  =  {F[0,0]*x_prior[0]:+.4f}")
    info(f"     x_pred[1]  =  {F[1,1]:.2f} × {x_prior[1]:+.4f}  =  {F[1,1]*x_prior[1]:+.4f}")
    x_pred = F @ x_prior
    print()
    print_matrix(x_pred.reshape(2, 1), "x_pred", prec=4, color=C.MAGENTA)

    print()
    section("Cálculo 2: P_pred = F · P_prior · Fᵀ + Q")
    P1 = F @ P_prior
    info("   Paso a:  F · P_prior  =")
    print_matrix(P1, color=C.BLUE)
    P2 = P1 @ F.T
    info("\n   Paso b:  (F · P_prior) · Fᵀ  =")
    print_matrix(P2, color=C.BLUE)
    P_pred = P2 + Q
    info("\n   Paso c:  + Q  =")
    print_matrix(P_pred, color=C.MAGENTA)

    print()
    note("→ Predicción terminada. P_pred crece (más incertidumbre) por sumar Q —\n"
         "  el mundo pudo haber 'pateado' el estado entre tick y tick.")
    wait()
    return x_pred, P_pred


def step_block_3b_update(x_pred, P_pred, z_norm, H, R):
    block_header("[BLOQUE 3b]", "FILTRO DE KALMAN — PASO CORRECCIÓN")
    info("'Ahora que veo la observación nueva, ¿cuánto me equivoqué? Combino mi\n"
         "predicción con la observación, dándole más peso a quien menos ruido tenga.'")

    section("Antes de calcular, conozcamos las matrices que faltan")

    # ---- H ----
    print_matrix_intro(
        "H",
        "matriz de observación. Filas = features (3), columnas = estados ocultos (2).\n"
        "       H[i,j] = 1 si la feature i 'mira' directamente al estado j; 0 si no.",
        H,
        prec=2,
        annotations=[
            (0, 0, "depth_imbalance OBSERVA la presión"),
            (0, 1, "depth_imbalance NO observa el régimen"),
            (1, 0, "ofi_l1 OBSERVA la presión"),
            (1, 1, "ofi_l1 NO observa el régimen"),
            (2, 0, "vol_30s NO observa la presión"),
            (2, 1, "vol_30s OBSERVA el régimen"),
        ],
    )

    # ---- R ----
    print_matrix_intro(
        "R",
        "covarianza del ruido de observación. Diagonal = qué tan ruidosa es cada\n"
        "       feature. Más alto = el filtro confía MENOS en esa feature.",
        R,
        annotations=[
            (0, 0, "varianza empírica de depth_imbalance, medida sobre 39k ticks limpios"),
            (1, 1, "varianza empírica de ofi_l1 (5× la de depth_imbalance — más ruidosa)"),
            (2, 2, "1.0 por construcción: tras estandarizar vol_30s tiene varianza = 1"),
            (0, 1, "0 → asumimos que los ruidos de las features son independientes entre sí"),
        ],
    )
    wait()

    section("Innovación: y = z_norm − H · x_pred")
    info("La innovación es 'cuánto se equivocó la predicción cuando llegó el dato real'.")
    Hx = H @ x_pred
    info("\n   H · x_pred =")
    print_matrix(Hx.reshape(3, 1), color=C.BLUE)
    y = z_norm - Hx
    print_matrix_intro(
        "y",
        "vector de innovación (residuo). Cada componente dice cuánto erró la\n"
        "       predicción para esa feature.",
        y.reshape(3, 1),
        annotations=[
            (0, None, "sorpresa en depth_imbalance vs predicción"),
            (1, None, "sorpresa en ofi_l1 vs predicción"),
            (2, None, "sorpresa en vol_30s_normalizada vs predicción"),
        ],
        color=C.MAGENTA,
    )
    feat_names = ["depth_imbalance", "ofi_l1", "vol_30s_norm"]
    biggest = int(np.argmax(np.abs(y)))
    note(f"\n→ La mayor sorpresa fue '{feat_names[biggest]}' "
         f"(|y|={abs(y[biggest]):.4f}).")
    wait()

    section("Covarianza de la innovación: S = H · P_pred · Hᵀ + R")
    info("S contesta: '¿qué tan grande esperaría yo que sea el residuo y, si todo va\n"
         "bien?' — el filtro lo usa para decidir cuánto confiar en cada feature.")
    HP = H @ P_pred
    HPH = HP @ H.T
    S = HPH + R
    info("\n   H · P_pred · Hᵀ =  (proyección de la incertidumbre del estado al espacio de obs)")
    print_matrix(HPH, color=C.BLUE)
    info("\n   + R =  (sumamos el ruido de las features)")
    print_matrix(R, color=C.BLUE)
    print_matrix_intro(
        "S",
        "covarianza de la innovación. Filas/columnas = features observadas (3×3).\n"
        "       Diagonal = cuánta varianza esperás en cada residuo.",
        S,
        annotations=[
            (0, 0, "incertidumbre esperada en y[depth_imbalance]"),
            (1, 1, "incertidumbre esperada en y[ofi_l1] — la más alta = la menos confiable"),
            (2, 2, "incertidumbre esperada en y[vol_30s]"),
            (0, 1, "covarianza cruzada — pequeña, pero distinta de cero porque ambas miran la misma 'presión'"),
        ],
        color=C.MAGENTA,
    )
    wait()

    section("Ganancia de Kalman: K = P_pred · Hᵀ · S⁻¹")
    info("K es la fórmula que combina predicción y observación. Te dice cuánto de\n"
         "cada residuo y se aplica a cada estado oculto.")
    PHt = P_pred @ H.T
    Sinv = np.linalg.inv(S)
    K = PHt @ Sinv
    info("\n   P_pred · Hᵀ =")
    print_matrix(PHt, color=C.BLUE)
    info("\n   S⁻¹ =  (la inversa de la covarianza de innovación)")
    print_matrix(Sinv, color=C.BLUE)
    print_matrix_intro(
        "K",
        "ganancia de Kalman (2×3). Filas = estados (presión, régimen), columnas =\n"
        "       features. K[i,j] = qué fracción de la sorpresa en la feature j se aplica\n"
        "       al estado i.",
        K,
        annotations=[
            (0, 0, f"{K[0,0]:.4f} → fracción de la sorpresa de depth_imbalance que mueve la PRESIÓN"),
            (0, 1, f"{K[0,1]:.4f} → fracción de la sorpresa de ofi_l1 que mueve la PRESIÓN"),
            (0, 2, f"{K[0,2]:.4f} → vol_30s NO mueve a la presión (≈0 por la estructura de H)"),
            (1, 0, f"{K[1,0]:.4f} → depth_imbalance NO mueve al régimen (≈0)"),
            (1, 1, f"{K[1,1]:.4f} → ofi_l1 NO mueve al régimen (≈0)"),
            (1, 2, f"{K[1,2]:.4f} → fracción de la sorpresa de vol_30s que mueve el RÉGIMEN"),
        ],
        color=C.MAGENTA,
    )
    note("\n→ La 'cruz de ceros' refleja H: depth/ofi sólo informan presión, vol_30s\n"
         "  sólo informa régimen. El filtro respeta esa separación automáticamente.")
    wait()

    section("Estado posterior: x_post = x_pred + K · y")
    info("Aplicamos la corrección. La predicción se 'mueve' por una cantidad\n"
         "proporcional al residuo y, escalado por la ganancia K.")
    Ky = K @ y
    info("\n   K · y =  (cuánto mover el estado en respuesta a las sorpresas)")
    print_matrix(Ky.reshape(2, 1), color=C.BLUE)
    x_post = x_pred + Ky
    print_matrix_intro(
        "x_post",
        "vector de estado posterior (2×1). Es lo que el filtro 'cree' ahora,\n"
        "       después de combinar predicción + observación.",
        x_post.reshape(2, 1),
        annotations=[
            (0, None, f"presión estimada FINAL del tick — pasa al bucket lookup"),
            (1, None, f"régimen estimado FINAL del tick"),
        ],
        color=C.MAGENTA,
    )
    print()
    highlight(f"  ⇒ presión estimada    = {x_post[0]:+.6f}")
    highlight(f"  ⇒ régimen estimado    = {x_post[1]:+.6f}")
    wait()

    section("Covarianza posterior: P_post = (I − K · H) · P_pred")
    info("Tras incorporar la observación, la incertidumbre baja. Esta es la nueva\n"
         "P que pasa al siguiente tick como P_prior.")
    KH = K @ H
    I_minus_KH = np.eye(2) - KH
    P_post = I_minus_KH @ P_pred
    info("\n   K · H =")
    print_matrix(KH, color=C.BLUE)
    info("\n   I − K · H =")
    print_matrix(I_minus_KH, color=C.BLUE)
    print_matrix_intro(
        "P_post",
        "covarianza posterior (2×2). Cuánta incertidumbre quedó después de absorber\n"
        "       la observación nueva. Pasa al siguiente tick como P_prior.",
        P_post,
        annotations=[
            (0, 0, "varianza FINAL de la presión (siempre ≤ varianza del prior)"),
            (1, 1, "varianza FINAL del régimen"),
            (0, 1, "covarianza presión↔régimen (sigue ≈0 por la estructura del modelo)"),
        ],
        color=C.MAGENTA,
    )
    note("\n→ Comparada con P_prior, la varianza BAJÓ — eso significa que el filtro está\n"
         "  más seguro de su estimación tras ver la observación.")
    wait()

    return x_post, P_post


def step_block_4_bucket(pressure: float):
    block_header("[BLOQUE 4]", "LOOKUP DEL BUCKET DE PRESIÓN")
    info("'¿En qué región del estado del mercado estamos ahora?'")
    print()
    section(f"Valor de presión: {pressure:+.6f}")
    section(f"Cortes (pressure_edges): {PRESSURE_EDGES}")
    print()
    bucket = which_bucket(pressure)
    info("Buckets (de izquierda a derecha):")
    for i in range(6):
        marker = C.MAGENTA + C.BOLD + " ← AQUÍ" + C.RESET if i == bucket else ""
        print(f"    Bucket {i}: {bucket_range_str(i):<22}  ({bucket_meaning(i)}){marker}")
    print()
    highlight(f"  ⇒ Bucket seleccionado: {bucket}  ({bucket_meaning(bucket)})")
    wait()
    return bucket


def step_block_5a_weights(bucket: int, frozen):
    block_header("[BLOQUE 5a]", "PESOS CONGELADOS DEL BUCKET")
    info("'Esta es la política APRENDIDA durante el entrenamiento offline.\n"
         "Para este bucket en particular, ¿qué probabilidad le doy a cada acción?'")
    print()
    visits = frozen["buckets"][bucket]["visits"]
    weights = frozen["buckets"][bucket]["weights"]
    section(f"Bucket {bucket} — entrenado con {visits:,} visitas en S{frozen['session_idx']}")
    print()
    actions = ["WAIT", "PASSIVE", "AGGRESSIVE"]
    colors  = {"WAIT": C.YELLOW, "PASSIVE": C.RED, "AGGRESSIVE": C.GREEN}
    print(f"    {'acción':<14}{'probabilidad':<16}distribución")
    print(f"    {'─'*12}  {'─'*12}  {'─'*44}")
    for a in actions:
        w = weights[a]
        bar_len = int(round(w * 50))
        bar = colors[a] + "█" * bar_len + C.RESET
        print(f"    {a:<14}{w:<16.4f}{bar} {w*100:.2f} %")
    print()
    print(f"    {'suma':<14}{sum(weights.values()):<16.4f}(debe ser 1)")
    wait()
    return weights


def step_block_5b_sample(weights, seed: int = 0):
    block_header("[BLOQUE 5b]", "MUESTREO DE LA ACCIÓN")
    info("'Tiro un dado de tres caras pesado por las probabilidades del bucket.'")
    print()
    actions = ["WAIT", "PASSIVE", "AGGRESSIVE"]
    probs   = [weights[a] for a in actions]
    cumsum  = np.cumsum(probs)

    section("Slices acumulados de la distribución")
    print(f"    [0.0000, {cumsum[0]:.4f}]  →  WAIT")
    print(f"    ({cumsum[0]:.4f}, {cumsum[1]:.4f}]  →  PASSIVE")
    print(f"    ({cumsum[1]:.4f}, {cumsum[2]:.4f}]  →  AGGRESSIVE")
    print()

    rng = np.random.default_rng(seed)
    u = float(rng.random())
    idx = int(np.searchsorted(cumsum, u))
    chosen = actions[idx]

    section("Tirar el dado")
    info(f"   uniform(0, 1)  →  u = {u:.4f}")
    info(f"   buscar en qué slice cae:  u={u:.4f} cae en el slice de {chosen}")
    print()
    color = {"WAIT": C.YELLOW, "PASSIVE": C.RED, "AGGRESSIVE": C.GREEN}[chosen]
    print(color + C.BOLD + f"  ⇒ Acción elegida: {chosen}" + C.RESET)
    wait()
    return chosen


def step_block_6_simulate(curr, nxt, action: str):
    block_header("[BLOQUE 6]", "SIMULACIÓN DEL FILL")
    info("'¿Qué hubiera pasado si realmente hubiéramos enviado esta orden a Binance?'")
    print()
    section("Estado del libro")
    kv("best_bid",   f"${curr['best_bid_px']:,.4f}")
    kv("best_ask",   f"${curr['best_ask_px']:,.4f}")
    kv("mid",        f"${curr['mid_price']:,.4f}")
    kv("next_mid",   f"${nxt['mid_price']:,.4f}")

    fill = simulate_fill(
        action=Action(action), ts_ms=curr["ts_ms"],
        curr_mid=curr["mid_price"],
        curr_best_bid=curr["best_bid_px"],
        curr_best_ask=curr["best_ask_px"],
        next_mid=nxt["mid_price"],
    )

    print()
    section(f"Acción: {action}")
    if action == "WAIT":
        info("   No envío ninguna orden. No hay fill.")
        kv("filled",     "FALSE", color_value=C.RED)
        kv("fill_price", "—")
    elif action == "PASSIVE":
        info("   Posteo una orden límite al best_bid.")
        info("   Se ejecuta sólo si en el próximo tick el mid baja (un vendedor cruza).")
        if fill.filled:
            info("   El próximo mid bajó → la orden se ejecutó.")
            good("   FILL al best_bid")
        else:
            info("   El próximo mid no bajó → la orden NO se ejecutó.")
            bad("   NO FILL")
        kv("filled",     "TRUE" if fill.filled else "FALSE",
           color_value=C.GREEN if fill.filled else C.RED)
        if fill.filled:
            kv("fill_price", f"${fill.fill_price:,.4f}", color_value=C.GREEN)
    else:
        info("   Cruzo el spread y compro a mercado al best_ask.")
        info("   Fill garantizado.")
        good("   FILL al best_ask")
        kv("filled",     "TRUE", color_value=C.GREEN)
        kv("fill_price", f"${fill.fill_price:,.4f}", color_value=C.GREEN)
    wait()
    return fill


def step_block_7_loss(fill, action: str):
    block_header("[BLOQUE 7]", "CÁLCULO DE LA PÉRDIDA")
    info("'¿Cuánto me costó esta acción en USDT?'")
    print()

    if fill.filled:
        slippage = fill.fill_price - fill.mid_price
        adverse  = max(0.0, fill.fill_price - fill.next_mid_price)
        info(f"   slippage  = fill_price − mid")
        info(f"             = ${fill.fill_price:,.4f} − ${fill.mid_price:,.4f}")
        if slippage >= 0:
            print(C.RED + f"             = +${slippage:.4f}  (PAGUÉ por encima del mid)" + C.RESET)
        else:
            print(C.GREEN + f"             = -${abs(slippage):.4f}  (compré por debajo del mid → GANANCIA)" + C.RESET)
        print()
        info(f"   adverse   = max(0, fill_price − next_mid)")
        info(f"             = max(0, ${fill.fill_price:,.4f} − ${fill.next_mid_price:,.4f})")
        if adverse > 0:
            print(C.RED + f"             = +${adverse:.4f}  (precio cayó después → mala compra)" + C.RESET)
        else:
            good(f"             = $0.0000  (precio no cayó → buena compra)")
    else:
        slippage = 0.0
        adverse  = max(0.0, fill.next_mid_price - fill.mid_price)
        info(f"   slippage  = 0  (no hubo fill)")
        print()
        info(f"   adverse   = max(0, next_mid − mid)")
        info(f"             = max(0, ${fill.next_mid_price:,.4f} − ${fill.mid_price:,.4f})")
        if adverse > 0:
            print(C.RED + f"             = +${adverse:.4f}  (precio subió → perdí oportunidad)" + C.RESET)
        else:
            good(f"             = $0.0000  (precio no subió → esperé bien)")
    print()
    L = slippage + LAMBDA * adverse
    info(f"   loss = slippage + λ · adverse_move")
    info(f"        = {slippage:+.4f} + {LAMBDA} × {adverse:.4f}")
    color = C.RED if L > 0 else C.GREEN
    print(color + C.BOLD + f"        = {L:+.6f}  ← pérdida del tick" + C.RESET)

    print()
    note("(El ahorro contra 'siempre-AGRESIVO' lo verás en el bloque RESUMEN al final\n"
         " del tick — se calcula corriendo en paralelo el baseline trivial.)")

    wait()
    return L, slippage, adverse


def step_block_8_hedge_update(weights, chosen: str, L: float):
    block_header("[BLOQUE 8]", "UPDATE DE HEDGE (ilustrativo)")
    info("'Si estuviéramos entrenando online, ¿cómo cambiarían los pesos del bucket\n"
         "después de observar esta pérdida?'")
    note("(En el GUI los pesos están CONGELADOS — ya entrenaron antes. Pero esto\n"
         "es lo que hizo el algoritmo durante el entrenamiento offline.)")
    print()

    factor = math.exp(-ETA * L)
    section(f"Factor multiplicativo: exp(−η · L) = exp(−{ETA} × {L:+.6f}) = {factor:.6f}")
    if factor < 1:
        info(f"   Factor < 1 → el peso de '{chosen}' se REDUCE (esa acción perdió plata).")
    elif factor > 1:
        info(f"   Factor > 1 → el peso de '{chosen}' AUMENTA (esa acción ganó plata).")
    else:
        info(f"   Factor = 1 → el peso no cambia.")
    print()

    section("Pesos antes y después de multiplicar (sin renormalizar)")
    actions = ["WAIT", "PASSIVE", "AGGRESSIVE"]
    new_un = dict(weights)
    new_un[chosen] = weights[chosen] * factor
    print(f"    {'acción':<14}{'antes':<14}{'después':<14}{'(cambio)'}")
    print(f"    {'─'*12}  {'─'*12}  {'─'*12}  {'─'*16}")
    for a in actions:
        d = new_un[a] - weights[a]
        col = C.WHITE if a != chosen else (C.RED if d < 0 else C.GREEN)
        print(f"    {a:<14}{weights[a]:<14.6f}{col}{new_un[a]:<14.6f}{C.RESET}({d:+.6f})")

    total_un = sum(new_un.values())
    print()
    section(f"Renormalización (suma actual = {total_un:.6f}, debe volver a 1)")
    new_n = {a: w / total_un for a, w in new_un.items()}
    print(f"    {'acción':<14}{'normalizado':<14}{'(Δ vs original)'}")
    print(f"    {'─'*12}  {'─'*12}  {'─'*16}")
    for a in actions:
        d = new_n[a] - weights[a]
        col = C.GREEN if d > 0 else (C.RED if d < 0 else C.WHITE)
        print(f"    {a:<14}{col}{new_n[a]:<14.6f}{C.RESET}({d:+.6f})")

    print()
    note("→ Con un solo tick el cambio es minúsculo (~0.0001). Pero después de\n"
         "  miles de ticks en este bucket, los cambios chiquitos compuestos producen\n"
         "  la distribución final que vimos en el bloque 5a.")
    wait()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session",    type=int, default=7,
                   help="Sesión a usar (1, 2, 3, 6, 7). Default: 7.")
    p.add_argument("--start-tick", type=int, default=800,
                   help="Tick desde el cual comenzar el walkthrough. Default: 800.")
    p.add_argument("--max-ticks",  type=int, default=3,
                   help="Cuántos ticks recorrer antes de salir. Default: 3.")
    p.add_argument("--seed",       type=int, default=0,
                   help="Semilla del RNG para muestreo. Default: 0.")
    p.add_argument("--auto",       action="store_true",
                   help="No esperar ENTER (modo prueba).")
    return p.parse_args()


def main():
    global AUTOPLAY
    args = parse_args()
    AUTOPLAY = args.auto

    # Bienvenida
    print()
    hr("═", C.YELLOW)
    print(C.YELLOW + C.BOLD + "  WALKTHROUGH INTERACTIVO — adversity-is-controllable" + C.RESET)
    hr("═", C.YELLOW)
    print()
    info("Recorrido paso a paso del pipeline completo: parquet → Kalman → bucket →")
    info("Hedge → simulador → pérdida. Presiona ENTER después de cada bloque para")
    info("avanzar.  Ctrl+C para salir.")
    print()
    note(f"Sesión:     S{args.session}")
    note(f"Tick inicial: {args.start_tick}")
    note(f"Ticks a recorrer: {args.max_ticks}")
    print()
    wait()

    # ----- carga de datos y warmup -----
    section("Cargando datos...")
    kalman_cfg = load_config("configs/kalman.yaml", KalmanConfig)
    fz_path = next(Path("data/derived/frozen_weights").glob("1d_session*_seed*.json"))
    with open(fz_path) as f:
        frozen = json.load(f)

    rows = list(iter_parquet_dir(Path("data/derived/features")))
    sessions = detect_sessions(rows)
    if args.session > len(sessions):
        bad(f"Sesión {args.session} no detectada (sólo hay {len(sessions)}).")
        return

    a, b = sessions[args.session - 1]
    sub = rows[a:b]
    if len(sub) < args.start_tick + args.max_ticks + 1:
        bad(f"La sesión {args.session} no tiene suficientes ticks "
            f"({len(sub)} disponibles, se piden {args.start_tick + args.max_ticks + 1}).")
        return

    # Replay Kalman desde el tick 0 hasta start_tick - 1 para tener el prior listo
    info(f"Calentando el filtro de Kalman (replay tick 0 → {args.start_tick - 1})...")
    kalman = KalmanFilter(kalman_cfg)
    obs = list(kalman_cfg.obs_features)
    prev = None
    for i in range(args.start_tick):
        row = sub[i]
        if prev is None:
            prev = row; continue
        z = np.array([prev[f] for f in obs], dtype=float)
        kalman.step(prev["ts_ms"], z)
        prev = row
    good(f"Listo. Ahora arrancamos el walkthrough en el tick {args.start_tick}.")

    # ----- loop -----
    F = kalman.F.copy()
    H = kalman.H.copy()
    Q = kalman.Q.copy()
    R = kalman.R.copy()

    cum_loss = 0.0
    cum_aggr = 0.0

    for offset in range(args.max_ticks):
        tick = args.start_tick + offset
        curr = sub[tick]
        nxt  = sub[tick + 1]

        print()
        hr("═", C.MAGENTA)
        print(C.MAGENTA + C.BOLD + f"  ━━━ TICK #{tick} (offset {offset+1}/{args.max_ticks}) ━━━" + C.RESET)
        hr("═", C.MAGENTA)

        # capturamos el prior antes de procesar este tick
        x_prior = kalman._x.copy()
        P_prior = kalman._P.copy()

        # bloques 1-2
        step_block_1_parquet(curr, nxt, tick)
        z_norm = step_block_2_standardize(curr, kalman_cfg)

        # bloque 3a (predict)
        x_pred, P_pred = step_block_3a_predict(x_prior, P_prior, F, Q)

        # bloque 3b (update)
        x_post, P_post = step_block_3b_update(x_pred, P_pred, z_norm, H, R)

        # actualizamos el estado del filtro real para el siguiente tick
        kalman._x = x_post.copy()
        kalman._P = P_post.copy()

        # bloque 4
        bucket = step_block_4_bucket(x_post[0])

        # bloques 5a, 5b
        weights = step_block_5a_weights(bucket, frozen)
        chosen  = step_block_5b_sample(weights, seed=args.seed + offset)

        # bloque 6
        fill = step_block_6_simulate(curr, nxt, chosen)

        # bloque 7
        L, slip, adv = step_block_7_loss(fill, chosen)
        cum_loss += L

        # baseline siempre-agresivo (para contar ahorros acumulados)
        aggr_slip = curr["best_ask_px"] - curr["mid_price"]
        aggr_adv  = max(0.0, curr["best_ask_px"] - nxt["mid_price"])
        aggr_L    = aggr_slip + LAMBDA * aggr_adv
        cum_aggr += aggr_L

        # bloque 8
        step_block_8_hedge_update(weights, chosen, L)

        # resumen acumulado
        block_header("[RESUMEN]", f"DESPUÉS DEL TICK {tick}")
        kv("Pérdida del tick",        f"${L:+.6f}",
           color_value=C.RED if L > 0 else C.GREEN)
        kv("Pérdida acumulada",       f"${cum_loss:+.4f}",
           color_value=C.RED if cum_loss > 0 else C.GREEN)
        kv("Always-AGG acumulado",    f"${cum_aggr:+.4f}", color_value=C.WHITE)
        savings = cum_aggr - cum_loss
        kv("Ahorro vs always-AGG",    f"${savings:+.4f}",
           color_value=C.GREEN if savings >= 0 else C.RED)
        wait()

    print()
    hr("═", C.GREEN)
    print(C.GREEN + C.BOLD + "  Fin del walkthrough — ¡ya conoces el sistema por dentro!" + C.RESET)
    hr("═", C.GREEN)


if __name__ == "__main__":
    main()
