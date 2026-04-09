#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
distortion_score_mcd.py

Replica práctica del análisis de "distortion score" del paper:
- entrena una distribución de referencia con entornos locales del bulk
- calcula por átomo una distancia robusta MCD (d_RB)
- clasifica outliers / defectos
- exporta CSV con score por átomo
- opcionalmente exporta XYZ extendido con la propiedad dRB

Dependencias:
    pip install numpy pandas scipy scikit-learn ase dscribe

Ejemplo:
    python distortion_score_mcd.py \
        --train bulk_perfect.dump \
        --test dump_defectuoso.dump \
        --species Ni \
        --format lammps-dump-text \
        --rcut 5.5 \
        --nmax 8 \
        --lmax 6 \
        --sigma 0.4 \
        --contamination 0.07 \
        --threshold-mode chi2 \
        --alpha 0.975 \
        --strata "4.0,12.0,17.0" \
        --out-prefix ni_test

Notas:
- El paper usa robust MCD distance d_RB como distortion score.
- Aquí se calcula a partir de la distancia de Mahalanobis robusta del modelo MinCovDet.
- sklearn devuelve distancia de Mahalanobis al cuadrado; por eso usamos sqrt().
- El paper usa bispectrum SO(4); aquí usamos SOAP para una implementación Python práctica.
"""

from __future__ import annotations
from sklearn.decomposition import PCA
import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from ase import Atoms
from ase.io import read, write
from dscribe.descriptors import SOAP
from scipy.stats import chi2
from sklearn.covariance import MinCovDet


# -----------------------------
# Utilidades
# -----------------------------
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def validate_file(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")


def parse_species(species_str: str) -> List[str]:
    species = [s.strip() for s in species_str.split(",") if s.strip()]
    if not species:
        raise ValueError("Debes indicar al menos una especie, por ejemplo: Ni o Fe")
    return species


# -----------------------------
# Remapeo de especies
# -----------------------------
def remap_atoms_species(atoms: Atoms, species: List[str]) -> Atoms:
    """
    Fuerza el mapeo de especies químicas luego de leer con ASE.

    Casos:
    - Si species tiene un solo elemento (ej. Ni), todos los átomos pasan a ser Ni.
    - Si hay varias especies, intenta usar el array 'type' de ASE.
    """
    atoms = atoms.copy()

    if len(species) == 1:
        atoms.set_chemical_symbols([species[0]] * len(atoms))
        return atoms

    arrays = atoms.arrays
    if "type" not in arrays:
        raise RuntimeError(
            "El archivo contiene múltiples especies pero no existe el array 'type' "
            "para mapearlas correctamente."
        )

    types = np.asarray(arrays["type"]).astype(int)
    unique_types = sorted(np.unique(types))

    if len(species) < max(unique_types):
        raise RuntimeError(
            f"No alcanza --species={species} para mapear tipos LAMMPS {unique_types}"
        )

    type_to_symbol = {t: species[t - 1] for t in unique_types}
    symbols = [type_to_symbol[t] for t in types]
    atoms.set_chemical_symbols(symbols)

    return atoms


# -----------------------------
# Lectura manual de dumps LAMMPS
# -----------------------------
def _parse_lammps_dump_manual(path: str, species: List[str], index: int = -1) -> Atoms:
    """
    Parser manual para dumps de LAMMPS que soporta:
    - BOX BOUNDS pp pp pp
    - BOX BOUNDS abc origin pp pp pp
    - columnas ATOMS con al menos: id type x y z

    Lee un solo frame. Si hay varios frames:
    - index=-1 -> último
    - index=0  -> primero
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.rstrip("\n") for line in f]

    frame_starts = [i for i, line in enumerate(lines) if line.startswith("ITEM: TIMESTEP")]
    if not frame_starts:
        raise RuntimeError(f"No se encontraron frames LAMMPS en: {path}")

    if index == -1:
        start = frame_starts[-1]
    elif index >= 0:
        if index >= len(frame_starts):
            raise IndexError(f"index={index} fuera de rango. Frames disponibles: {len(frame_starts)}")
        start = frame_starts[index]
    else:
        raise ValueError("Solo se soporta index=-1 o index>=0")

    next_candidates = [i for i in frame_starts if i > start]
    end = next_candidates[0] if next_candidates else len(lines)
    chunk = lines[start:end]

    ptr = 0

    if not chunk[ptr].startswith("ITEM: TIMESTEP"):
        raise RuntimeError("Frame inválido: falta ITEM: TIMESTEP")
    ptr += 2

    if not chunk[ptr].startswith("ITEM: NUMBER OF ATOMS"):
        raise RuntimeError("Frame inválido: falta ITEM: NUMBER OF ATOMS")
    natoms = int(chunk[ptr + 1].strip())
    ptr += 2

    if not chunk[ptr].startswith("ITEM: BOX BOUNDS"):
        raise RuntimeError("Frame inválido: falta ITEM: BOX BOUNDS")

    box_header = chunk[ptr]
    ptr += 1

    if "abc origin" not in box_header:
        xline = [float(x) for x in chunk[ptr].split()]
        yline = [float(x) for x in chunk[ptr + 1].split()]
        zline = [float(x) for x in chunk[ptr + 2].split()]
        ptr += 3

        if len(xline) < 2 or len(yline) < 2 or len(zline) < 2:
            raise RuntimeError("BOX BOUNDS ortogonal inválido")

        xlo, xhi = xline[0], xline[1]
        ylo, yhi = yline[0], yline[1]
        zlo, zhi = zline[0], zline[1]

        cell = np.array(
            [
                [xhi - xlo, 0.0, 0.0],
                [0.0, yhi - ylo, 0.0],
                [0.0, 0.0, zhi - zlo],
            ],
            dtype=float,
        )

    else:
        a_line = [float(x) for x in chunk[ptr].split()]
        b_line = [float(x) for x in chunk[ptr + 1].split()]
        c_line = [float(x) for x in chunk[ptr + 2].split()]
        ptr += 3

        if len(a_line) != 4 or len(b_line) != 4 or len(c_line) != 4:
            raise RuntimeError("Formato 'abc origin' inválido; se esperaban 4 números por línea")

        cell = np.array(
            [
                [a_line[0], a_line[1], a_line[2]],
                [b_line[0], b_line[1], b_line[2]],
                [c_line[0], c_line[1], c_line[2]],
            ],
            dtype=float,
        )

    if not chunk[ptr].startswith("ITEM: ATOMS"):
        raise RuntimeError("Frame inválido: falta ITEM: ATOMS")

    atom_header = chunk[ptr].split()[2:]
    ptr += 1

    required = ["id", "type", "x", "y", "z"]
    for col in required:
        if col not in atom_header:
            raise RuntimeError(f"Falta columna requerida '{col}' en ITEM: ATOMS")

    col_id = atom_header.index("id")
    col_type = atom_header.index("type")
    col_x = atom_header.index("x")
    col_y = atom_header.index("y")
    col_z = atom_header.index("z")

    ids = np.empty(natoms, dtype=int)
    types = np.empty(natoms, dtype=int)
    positions = np.empty((natoms, 3), dtype=float)

    for i in range(natoms):
        parts = chunk[ptr + i].split()
        ids[i] = int(parts[col_id])
        types[i] = int(parts[col_type])
        positions[i, 0] = float(parts[col_x])
        positions[i, 1] = float(parts[col_y])
        positions[i, 2] = float(parts[col_z])

    order = np.argsort(ids)
    ids = ids[order]
    types = types[order]
    positions = positions[order]

    unique_types = sorted(np.unique(types))
    if len(species) == 1:
        type_to_symbol = {t: species[0] for t in unique_types}
    else:
        if len(species) < max(unique_types):
            raise RuntimeError(
                f"No alcanza --species={species} para mapear tipos LAMMPS {unique_types}"
            )
        type_to_symbol = {t: species[t - 1] for t in unique_types}

    symbols = [type_to_symbol[t] for t in types]

    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=cell,
        pbc=True,
    )

    atoms.arrays["id"] = ids
    atoms.arrays["type"] = types

    return atoms


def read_structure(
    path: str,
    file_format: Optional[str] = None,
    index: int = -1,
    species: Optional[List[str]] = None,
) -> Atoms:
    """
    Intenta leer con ASE. Si falla para dumps LAMMPS con 'abc origin',
    usa parser manual.

    Además, si el archivo es un dump LAMMPS y se pasa species,
    fuerza el mapeo correcto de símbolos químicos.
    """
    kwargs = {}
    if file_format is not None:
        kwargs["format"] = file_format

    try:
        atoms = read(path, index=index, **kwargs)
        if not isinstance(atoms, Atoms):
            raise RuntimeError(f"No se pudo leer una estructura única desde: {path}")

        is_lammps_dump = (file_format == "lammps-dump-text") or path.endswith(".dump")
        if is_lammps_dump and species is not None:
            atoms = remap_atoms_species(atoms, species)

        return atoms

    except Exception as e:
        msg = str(e)
        is_lammps_dump = (file_format == "lammps-dump-text") or path.endswith(".dump")

        if is_lammps_dump and ("'xy' is not in list" in msg or "abc origin" in msg):
            if species is None:
                raise RuntimeError(
                    "Falló ASE y se necesita species para el parser manual"
                ) from e
            eprint(f"[WARN] ASE no pudo leer {path}. Uso parser manual LAMMPS...")
            return _parse_lammps_dump_manual(path, species=species, index=index)

        raise


# -----------------------------
# Descriptor local
# -----------------------------
def build_soap(species: List[str], rcut: float, nmax: int, lmax: int, sigma: float) -> SOAP:
    return SOAP(
        species=species,
        periodic=True,
        r_cut=rcut,
        n_max=nmax,
        l_max=lmax,
        sigma=sigma,
        sparse=False,
        average="off",
    )


def compute_local_descriptors(atoms: Atoms, soap: SOAP,
                               chunk_size: int = 50000) -> np.ndarray:
    """
    Retorna matriz [N_atoms, N_features] con descriptor local por átomo.
    Procesa en chunks para evitar segfault por memoria en sistemas grandes.
    """
    n = len(atoms)
    if n <= chunk_size:
        return soap.create(atoms)

    # Primer chunk para conocer n_features
    first = soap.create(atoms, positions=list(range(min(chunk_size, n))))
    n_features = first.shape[1]
    out = np.empty((n, n_features), dtype=np.float64)
    out[:first.shape[0]] = first
    del first

    for start in range(chunk_size, n, chunk_size):
        end = min(start + chunk_size, n)
        eprint(f"[INFO]   SOAP chunk {start//chunk_size + 1}"
               f" ({start}-{end} de {n})...")
        out[start:end] = soap.create(atoms, positions=list(range(start, end)))

    return out


# -----------------------------
# Modelo MCD y scoring
# -----------------------------
@dataclass
class DistortionModel:
    mcd: MinCovDet
    threshold: float
    threshold_mode: str
    alpha: float
    contamination: float
    n_features: int

    def score(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self, "pca"):
            X = self.pca.transform(X)

        d2 = self.mcd.mahalanobis(X)
        return np.sqrt(np.maximum(d2, 0.0))

    def predict_outlier(self, X: np.ndarray) -> np.ndarray:
        d = self.score(X)
        return d > self.threshold

def fit_mcd_model(
    X_train: np.ndarray,
    contamination: float = 0.07,
    threshold_mode: str = "chi2",
    alpha: float = 0.975,
) -> DistortionModel:

    # -------- PCA para evitar matriz singular --------
    pca = PCA(n_components=0.99, svd_solver="full")  # conserva 99% varianza
    X_train = pca.fit_transform(X_train)

    print(f"[INFO] PCA -> nueva dimensión: {X_train.shape[1]}")

    # -------- MCD --------
    mcd = MinCovDet(
        support_fraction=0.95,  # ← clave
        random_state=42,
        assume_centered=False,
    )
    mcd.fit(X_train)

    d2_train = mcd.mahalanobis(X_train)
    d_train = np.sqrt(np.maximum(d2_train, 0.0))

    if threshold_mode == "chi2":
        threshold = math.sqrt(chi2.ppf(alpha, df=X_train.shape[1]))
    else:
        threshold = float(np.quantile(d_train, alpha))

    model = DistortionModel(
        mcd=mcd,
        threshold=threshold,
        threshold_mode=threshold_mode,
        alpha=alpha,
        contamination=contamination,
        n_features=X_train.shape[1],
    )

    model.pca = pca  # ← guardar PCA

    return model

# -----------------------------
# Export
# -----------------------------
def export_csv(atoms: Atoms, d_rb: np.ndarray, outlier_mask: np.ndarray, out_csv: str) -> None:
    pos = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    syms = atoms.get_chemical_symbols()

    df = pd.DataFrame(
        {
            "atom_index": np.arange(len(atoms)),
            "Z": numbers,
            "symbol": syms,
            "x": pos[:, 0],
            "y": pos[:, 1],
            "z": pos[:, 2],
            "dRB": d_rb,
            "is_outlier": outlier_mask.astype(int),
        }
    )
    df.to_csv(out_csv, index=False)


def export_extended_xyz(atoms: Atoms, d_rb: np.ndarray, outlier_mask: np.ndarray, out_xyz: str) -> None:
    atoms2 = atoms.copy()
    atoms2.set_array("dRB", np.asarray(d_rb, dtype=float))
    atoms2.set_array("is_outlier", np.asarray(outlier_mask, dtype=int))
    write(out_xyz, atoms2, format="extxyz")


def summarize_scores(d_rb: np.ndarray, outlier_mask: np.ndarray) -> dict:
    q = np.quantile(d_rb, [0.5, 0.9, 0.95, 0.99]).tolist()
    return {
        "n_atoms": int(len(d_rb)),
        "n_outliers": int(outlier_mask.sum()),
        "fraction_outliers": float(outlier_mask.mean()),
        "dRB_min": float(np.min(d_rb)),
        "dRB_max": float(np.max(d_rb)),
        "dRB_mean": float(np.mean(d_rb)),
        "dRB_median": float(q[0]),
        "dRB_q90": float(q[1]),
        "dRB_q95": float(q[2]),
        "dRB_q99": float(q[3]),
    }


# -----------------------------
# Visualización
# -----------------------------
def plot_dRB_histogram(d_rb_train: np.ndarray, d_rb_test: np.ndarray,
                       threshold: float, prefix: str) -> str:
    """Histograma comparativo train vs test con línea de umbral."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Train
    ax = axes[0]
    ax.hist(d_rb_train, bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(threshold, color="#C44E52", ls="--", lw=2, label=f"Umbral = {threshold:.2f}")
    n_out_train = int(np.sum(d_rb_train > threshold))
    ax.set_title(f"Train  (n={len(d_rb_train)}, outliers={n_out_train})", fontsize=12)
    ax.set_xlabel("d_RB (distortion score)")
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=9)

    # Test
    ax = axes[1]
    ax.hist(d_rb_test, bins=60, color="#DD8452", edgecolor="white", alpha=0.85)
    ax.axvline(threshold, color="#C44E52", ls="--", lw=2, label=f"Umbral = {threshold:.2f}")
    n_out_test = int(np.sum(d_rb_test > threshold))
    ax.set_title(f"Test  (n={len(d_rb_test)}, outliers={n_out_test})", fontsize=12)
    ax.set_xlabel("d_RB (distortion score)")
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=9)

    fig.suptitle("Distribución de Distortion Score (d_RB)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = f"{prefix}_hist_dRB.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_spatial_scatter(atoms: Atoms, d_rb: np.ndarray, threshold: float,
                         prefix: str, label: str = "test") -> str:
    """Scatter 3D de posiciones atómicas coloreadas por d_RB."""
    pos = atoms.get_positions()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                    c=d_rb, cmap="plasma", s=4, alpha=0.7)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cb.set_label("d_RB")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.set_title(f"Distortion Score espacial – {label}  (n={len(atoms)})", fontsize=12)

    path = f"{prefix}_spatial_{label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_2d_slices(atoms: Atoms, d_rb: np.ndarray, prefix: str,
                   n_slices: int = 4, label: str = "test") -> str:
    """Proyecciones 2D en cortes a lo largo de z, para ver la distribución espacial."""
    pos = atoms.get_positions()
    z_vals = pos[:, 2]
    z_edges = np.linspace(z_vals.min(), z_vals.max(), n_slices + 1)

    fig, axes = plt.subplots(1, n_slices, figsize=(4 * n_slices, 4), sharey=True)
    if n_slices == 1:
        axes = [axes]

    vmin, vmax = d_rb.min(), d_rb.max()

    for i in range(n_slices):
        mask = (z_vals >= z_edges[i]) & (z_vals < z_edges[i + 1])
        if i == n_slices - 1:
            mask = (z_vals >= z_edges[i]) & (z_vals <= z_edges[i + 1])

        ax = axes[i]
        sc = ax.scatter(pos[mask, 0], pos[mask, 1], c=d_rb[mask],
                        cmap="plasma", s=6, alpha=0.8, vmin=vmin, vmax=vmax)
        ax.set_title(f"z ∈ [{z_edges[i]:.1f}, {z_edges[i+1]:.1f}] Å", fontsize=10)
        ax.set_xlabel("x (Å)")
        if i == 0:
            ax.set_ylabel("y (Å)")
        ax.set_aspect("equal")

    fig.suptitle(f"Cortes XY coloreados por d_RB – {label}", fontsize=13, fontweight="bold")
    fig.colorbar(sc, ax=axes, shrink=0.8, label="d_RB")
    fig.tight_layout(rect=[0, 0, 0.92, 0.93])
    path = f"{prefix}_slices_{label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_stratification(strata_counts: dict, prefix: str) -> str:
    """Barras de conteo por estrato."""
    labels = list(strata_counts.keys())
    counts = list(strata_counts.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white")

    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                str(c), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Cantidad de átomos")
    ax.set_title("Estratificación por rango de d_RB", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    path = f"{prefix}_stratification.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_summary_dashboard(train_summary: dict, test_summary: dict,
                           model_info: dict, prefix: str) -> str:
    """Dashboard resumen con métricas clave en una sola figura."""
    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # --- Box plot comparativo ---
    ax0 = fig.add_subplot(gs[0, 0])
    bp = ax0.boxplot(
        [[train_summary["dRB_min"], train_summary["dRB_q90"],
          train_summary["dRB_median"], train_summary["dRB_q95"],
          train_summary["dRB_max"]],
         [test_summary["dRB_min"], test_summary["dRB_q90"],
          test_summary["dRB_median"], test_summary["dRB_q95"],
          test_summary["dRB_max"]]],
        labels=["Train", "Test"],
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("#4C72B0")
    bp["boxes"][1].set_facecolor("#DD8452")
    ax0.set_ylabel("d_RB")
    ax0.set_title("Distribución d_RB", fontsize=11)

    # --- Tabla de métricas ---
    ax1 = fig.add_subplot(gs[0, 1:])
    ax1.axis("off")
    row_labels = ["n_atoms", "n_outliers", "frac_outliers",
                  "dRB_min", "dRB_mean", "dRB_median",
                  "dRB_q90", "dRB_q95", "dRB_max"]
    cell_text = []
    for k in row_labels:
        tv = train_summary.get(k, "—")
        ev = test_summary.get(k, "—")
        if isinstance(tv, float):
            tv = f"{tv:.4f}"
        if isinstance(ev, float):
            ev = f"{ev:.4f}"
        cell_text.append([str(tv), str(ev)])

    table = ax1.table(cellText=cell_text, rowLabels=row_labels,
                      colLabels=["Train", "Test"], loc="center",
                      cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    ax1.set_title("Métricas Resumen", fontsize=11, pad=20)

    # --- Parámetros del modelo ---
    ax2 = fig.add_subplot(gs[1, :])
    ax2.axis("off")
    info_text = (
        f"SOAP: r_cut={model_info['soap']['r_cut']}, "
        f"n_max={model_info['soap']['n_max']}, "
        f"l_max={model_info['soap']['l_max']}, "
        f"σ={model_info['soap']['sigma']}    |    "
        f"MCD: contamination={model_info['mcd']['contamination']}, "
        f"threshold_mode={model_info['mcd']['threshold_mode']}, "
        f"α={model_info['mcd']['alpha']}, "
        f"threshold_dRB={model_info['mcd']['threshold_dRB']:.4f}, "
        f"n_features_PCA={model_info['mcd']['n_features']}"
    )
    ax2.text(0.5, 0.5, info_text, ha="center", va="center", fontsize=10,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f0f0", edgecolor="#999"))

    fig.suptitle("Dashboard – Distortion Score MCD", fontsize=15, fontweight="bold")
    path = f"{prefix}_dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Calcula distortion score por átomo con MCD sobre descriptores SOAP."
    )
    parser.add_argument("--train", required=True, help="Estructura bulk/reference")
    parser.add_argument("--test", required=True, help="Estructura a analizar")
    parser.add_argument("--format", default=None, help="Formato ASE, ej: lammps-dump-text")
    parser.add_argument("--species", required=True, help="Especies separadas por coma, ej: Ni o Fe,C")
    parser.add_argument("--rcut", type=float, default=5.5, help="Cutoff SOAP")
    parser.add_argument("--nmax", type=int, default=8, help="n_max SOAP")
    parser.add_argument("--lmax", type=int, default=6, help="l_max SOAP")
    parser.add_argument("--sigma", type=float, default=0.4, help="sigma SOAP")
    parser.add_argument("--contamination", type=float, default=0.07, help="contamination / outlier prior")
    parser.add_argument(
        "--threshold-mode",
        choices=["chi2", "empirical"],
        default="chi2",
        help="Modo de umbral",
    )
    parser.add_argument("--alpha", type=float, default=0.975, help="Nivel para umbral")
    parser.add_argument("--out-prefix", default="distortion_score", help="Prefijo de salida")
    parser.add_argument(
        "--strata",
        default="",
        help="Umbrales extra para estratificación, ej: '4.0,12.0,17.0'",
    )

    args = parser.parse_args()

    validate_file(args.train)
    validate_file(args.test)

    species = parse_species(args.species)

    eprint("[INFO] Leyendo estructuras...")
    train_atoms = read_structure(args.train, args.format, species=species)
    test_atoms = read_structure(args.test, args.format, species=species)

    if len(species) == 1:
        train_atoms.set_chemical_symbols([species[0]] * len(train_atoms))
        test_atoms.set_chemical_symbols([species[0]] * len(test_atoms))

    eprint(f"[INFO] Train atoms: {len(train_atoms)}")
    eprint(f"[INFO] Test atoms : {len(test_atoms)}")
    eprint(f"[INFO] Train symbols únicos: {sorted(set(train_atoms.get_chemical_symbols()))}")
    eprint(f"[INFO] Test symbols únicos : {sorted(set(test_atoms.get_chemical_symbols()))}")
    eprint(f"[INFO] Train Z únicos      : {sorted(set(train_atoms.get_atomic_numbers()))}")
    eprint(f"[INFO] Test Z únicos       : {sorted(set(test_atoms.get_atomic_numbers()))}")

    eprint("[INFO] Construyendo descriptor SOAP...")
    soap = build_soap(
        species=species,
        rcut=args.rcut,
        nmax=args.nmax,
        lmax=args.lmax,
        sigma=args.sigma,
    )

    eprint("[INFO] Calculando descriptores locales...")
    X_train = compute_local_descriptors(train_atoms, soap)
    X_test = compute_local_descriptors(test_atoms, soap)

    eprint(f"[INFO] X_train shape = {X_train.shape}")
    eprint(f"[INFO] X_test  shape = {X_test.shape}")

    eprint("[INFO] Ajustando MinCovDet...")
    model = fit_mcd_model(
        X_train,
        contamination=args.contamination,
        threshold_mode=args.threshold_mode,
        alpha=args.alpha,
    )

    eprint(f"[INFO] threshold dRB = {model.threshold:.6f}")

    d_rb_train = model.score(X_train)
    d_rb_test = model.score(X_test)
    outlier_mask_test = d_rb_test > model.threshold

    train_summary = summarize_scores(d_rb_train, d_rb_train > model.threshold)
    test_summary = summarize_scores(d_rb_test, outlier_mask_test)

    out_csv = f"{args.out_prefix}_scores.csv"
    out_xyz = f"{args.out_prefix}_scored.extxyz"
    out_json = f"{args.out_prefix}_summary.json"

    export_csv(test_atoms, d_rb_test, outlier_mask_test, out_csv)
    export_extended_xyz(test_atoms, d_rb_test, outlier_mask_test, out_xyz)

    result = {
        "train_file": args.train,
        "test_file": args.test,
        "species": species,
        "soap": {
            "r_cut": args.rcut,
            "n_max": args.nmax,
            "l_max": args.lmax,
            "sigma": args.sigma,
        },
        "mcd": {
            "contamination": args.contamination,
            "threshold_mode": args.threshold_mode,
            "alpha": args.alpha,
            "threshold_dRB": model.threshold,
            "n_features": model.n_features,
        },
        "train_summary": train_summary,
        "test_summary": test_summary,
    }

    if args.strata.strip():
        strata = [float(x.strip()) for x in args.strata.split(",") if x.strip()]
        strata = sorted(strata)
        bins = [-np.inf] + strata + [np.inf]
        labels = []
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            if np.isneginf(lo):
                labels.append(f"dRB<= {hi}")
            elif np.isposinf(hi):
                labels.append(f"dRB> {lo}")
            else:
                labels.append(f"{lo} < dRB <= {hi}")

        strata_ids = pd.cut(d_rb_test, bins=bins, labels=False, include_lowest=True)
        strata_counts = {}
        for i, lab in enumerate(labels):
            strata_counts[lab] = int(np.sum(strata_ids == i))

        result["stratification"] = {
            "thresholds": strata,
            "counts": strata_counts,
        }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # -------- Visualizaciones --------
    eprint("[INFO] Generando visualizaciones...")
    plots = []

    plots.append(plot_dRB_histogram(d_rb_train, d_rb_test, model.threshold, args.out_prefix))
    plots.append(plot_spatial_scatter(test_atoms, d_rb_test, model.threshold,
                                     args.out_prefix, label="test"))
    plots.append(plot_2d_slices(test_atoms, d_rb_test, args.out_prefix,
                                n_slices=4, label="test"))

    if "stratification" in result:
        plots.append(plot_stratification(result["stratification"]["counts"], args.out_prefix))

    plots.append(plot_summary_dashboard(train_summary, test_summary, result, args.out_prefix))

    print("\n=== RESULTADOS ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n[OK] CSV   : {out_csv}")
    print(f"[OK] XYZ   : {out_xyz}")
    print(f"[OK] JSON  : {out_json}")
    for p in plots:
        print(f"[OK] PLOT  : {p}")


if __name__ == "__main__":
    main()
