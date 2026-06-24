"""
=============================================================
TESS Exoplanet Pipeline — Bharatiya Antariksh Hackathon 2026
Problem Statement 7: AI-enabled Detection of Exoplanets
=============================================================

What this script does:
  1. Downloads ExoFOP-TESS TOI catalog (labeled: planet/EB/false positive/other)
  2. Downloads TESS-EB catalog (Prša et al. 2022) for eclipsing binary TIC IDs
  3. Fetches TESS PDCSAP_FLUX light curves via lightkurve for each TIC ID
  4. Extracts time-series features per star
  5. Saves a final CSV with all features + labels

Output: tess_exoplanet_dataset.csv
  - One row per star
  - Columns: TIC_ID, label (4 classes), flux time series features,
             period, transit_depth, transit_duration, SNR, etc.

Requirements:
  pip install lightkurve astroquery astropy pandas numpy requests scipy

Run:
  python tess_pipeline.py
"""

import os
import time as time_module
import warnings
import requests
import numpy as np
import pandas as pd
from io import StringIO
from scipy import signal
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
N_PER_CLASS   = 50          # stars to fetch per class (increase for more data)
CADENCE       = "short"     # "short" = 2-min TESS cadence
OUTPUT_CSV    = "tess_exoplanet_dataset.csv"
FLUX_POINTS   = 1000        # resample every light curve to this fixed length

# Class labels (matching PS-7 classification framework)
LABEL_MAP = {
    "transit"  : 0,   # confirmed exoplanet transit
    "eclipse"  : 1,   # eclipsing binary
    "blend"    : 2,   # blended / background EB (false positive)
    "other"    : 3,   # non-transiting / stellar variability / noise
}

# ── STEP 1: Download ExoFOP TOI Catalog ───────────────────────────────────────
def fetch_exofop_catalog():
    """
    Downloads the TESS Object of Interest (TOI) catalog from ExoFOP.
    Contains confirmed planets, false positives, and eclipsing binaries
    with their TIC IDs and dispositions.
    """
    print("\n[1/5] Downloading ExoFOP-TESS TOI catalog...")
    url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), comment="#")

    # Rename key columns for clarity
    df.columns = df.columns.str.strip()
    print(f"   TOI catalog loaded: {len(df)} entries")
    print(f"   Columns: {list(df.columns[:10])} ...")
    return df


def select_tic_ids_from_toi(toi_df, n_per_class=N_PER_CLASS):
    """
    Selects TIC IDs for each of the 4 target classes from the TOI catalog.

    ExoFOP dispositions:
      KP / CP   → Confirmed Planet         → label: transit
      FP        → False Positive (general) → label: blend
      EB        → Eclipsing Binary         → label: eclipse

    We also pick low-score candidates as 'other' (stellar variability / noise).
    """
    print("\n[2/5] Selecting TIC IDs per class...")
    tic_labels = []

    # Try common column name variants in ExoFOP CSV
    disp_col = None
    for col in ["TFOPWG Disposition", "tfopwg_disp", "Disposition", "disposition"]:
        if col in toi_df.columns:
            disp_col = col
            break

    tic_col = None
    for col in ["TIC ID", "tic_id", "TIC"]:
        if col in toi_df.columns:
            tic_col = col
            break

    if disp_col is None or tic_col is None:
        print(f"   WARNING: Could not find disposition/TIC columns.")
        print(f"   Available columns: {list(toi_df.columns)}")
        return []

    dispositions = toi_df[disp_col].str.strip().str.upper()

    # --- Transit class: Confirmed Planets ---
    confirmed = toi_df[dispositions.isin(["KP", "CP"])][tic_col].dropna().unique()
    for tic in confirmed[:n_per_class]:
        tic_labels.append({"TIC_ID": int(tic), "label": "transit", "label_int": LABEL_MAP["transit"]})

    # --- Eclipse class: Eclipsing Binaries ---
    # ExoFOP marks EBs as "EB" in TFOPWG disposition OR as FP with EB in comments
    eb = toi_df[dispositions.str.contains("EB", na=False)][tic_col].dropna().unique()
    for tic in eb[:n_per_class]:
        tic_labels.append({"TIC_ID": int(tic), "label": "eclipse", "label_int": LABEL_MAP["eclipse"]})

    # --- Blend class: False Positives ---
    fp = toi_df[dispositions == "FP"][tic_col].dropna().unique()
    for tic in fp[:n_per_class]:
        tic_labels.append({"TIC_ID": int(tic), "label": "blend", "label_int": LABEL_MAP["blend"]})

    # --- Other class: Low-confidence candidates (PC with low score) ---
    pc = toi_df[dispositions == "PC"][tic_col].dropna().unique()
    for tic in pc[:n_per_class]:
        tic_labels.append({"TIC_ID": int(tic), "label": "other", "label_int": LABEL_MAP["other"]})

    label_df = pd.DataFrame(tic_labels)
    for lbl, grp in label_df.groupby("label"):
        print(f"   {lbl:10s}: {len(grp)} TIC IDs selected")

    return label_df


# ── STEP 2: Fetch Light Curves ─────────────────────────────────────────────────
def fetch_light_curve(tic_id, cadence=CADENCE):
    """
    Fetches a TESS PDCSAP_FLUX light curve for a given TIC ID using lightkurve.
    Returns (time_array, flux_array) or (None, None) if unavailable.
    """
    import lightkurve as lk

    try:
        search = lk.search_lightcurve(
            f"TIC {tic_id}",
            mission="TESS",
            cadence=cadence,
            author="SPOC"         # official TESS pipeline (has PDCSAP_FLUX)
        )

        if len(search) == 0:
            # Fall back to 10-min cadence (TESS Extended Mission)
            search = lk.search_lightcurve(
                f"TIC {tic_id}",
                mission="TESS",
                cadence="fast",
                author="SPOC"
            )

        if len(search) == 0:
            return None, None, {}

        # Download the most recent sector
        lc = search[-1].download()
        lc = lc.remove_nans().remove_outliers(sigma=4)

        # Use PDCSAP_FLUX (systematics-corrected) — best for transit detection
        time = lc.time.value
        flux = lc.pdcsap_flux.value if hasattr(lc, "pdcsap_flux") else lc.flux.value

        # Normalize flux (subtract median, divide by median)
        flux = flux / np.nanmedian(flux) - 1.0

        # Collect metadata
        meta = {
            "sector"     : search[-1].mission[0] if hasattr(search[-1], "mission") else "?",
            "exptime_sec": search[-1].exptime.value[0] if hasattr(search[-1], "exptime") else 0,
            "n_points_raw": len(time),
        }

        return time, flux, meta

    except Exception as e:
        print(f"      Error fetching TIC {tic_id}: {e}")
        return None, None, {}


def resample_flux(lc_time, flux, n_points=FLUX_POINTS):
    """
    Resamples a light curve to a fixed number of evenly-spaced time points.
    This is needed to make all rows the same length for ML training.
    """
    t_new = np.linspace(lc_time.min(), lc_time.max(), n_points)
    flux_new = np.interp(t_new, lc_time, flux)
    return t_new, flux_new


# ── STEP 3: Extract Features ───────────────────────────────────────────────────
def extract_features(time, flux):
    """
    Extracts astrophysically meaningful features from a light curve.
    These features are used by the ML classifier AND to estimate
    transit parameters (depth, period, duration) as required by PS-7.
    """
    feats = {}

    # --- Statistical features ---
    feats["flux_mean"]     = np.mean(flux)
    feats["flux_std"]      = np.std(flux)
    feats["flux_min"]      = np.min(flux)
    feats["flux_max"]      = np.max(flux)
    feats["flux_range"]    = feats["flux_max"] - feats["flux_min"]
    feats["flux_skew"]     = skew(flux)
    feats["flux_kurtosis"] = kurtosis(flux)
    feats["flux_median"]   = np.median(flux)

    # --- Transit depth (minimum flux dip below median) ---
    feats["transit_depth"] = abs(feats["flux_min"] - feats["flux_median"])

    # --- Signal-to-Noise Ratio (SNR) ---
    noise = feats["flux_std"]
    feats["SNR"] = feats["transit_depth"] / noise if noise > 0 else 0

    # --- Periodicity via Lomb-Scargle periodogram ---
    try:
        from astropy.timeseries import LombScargle
        ls = LombScargle(time, flux)
        frequency, power = ls.autopower(minimum_frequency=1/30, maximum_frequency=1/0.1)
        periods = 1 / frequency

        best_idx = np.argmax(power)
        feats["ls_period"]      = periods[best_idx]
        feats["ls_peak_power"]  = power[best_idx]
        feats["ls_fap"]         = ls.false_alarm_probability(power[best_idx])  # significance
    except Exception:
        feats["ls_period"]      = 0
        feats["ls_peak_power"]  = 0
        feats["ls_fap"]         = 1.0

    # --- Transit duration estimate ---
    # Find contiguous regions below 3-sigma threshold
    threshold = feats["flux_median"] - 3 * feats["flux_std"]
    in_dip = flux < threshold
    if np.any(in_dip):
        dip_times = time[in_dip]
        feats["transit_duration_days"] = dip_times[-1] - dip_times[0] if len(dip_times) > 1 else 0
    else:
        feats["transit_duration_days"] = 0

    # --- Count number of dips (proxy for number of transits) ---
    dip_indices = np.where(np.diff(in_dip.astype(int)) == 1)[0]
    feats["n_dips"] = len(dip_indices)

    # --- Even/odd depth ratio (eclipsing binary diagnostic) ---
    # EBs show alternating deep/shallow eclipses; planets don't
    if feats["n_dips"] >= 2:
        dip_depths = []
        for idx in dip_indices:
            window = flux[max(0, idx-5): idx+5]
            dip_depths.append(abs(np.min(window)))
        even_depths = dip_depths[::2]
        odd_depths  = dip_depths[1::2]
        if odd_depths:
            feats["even_odd_ratio"] = np.mean(even_depths) / (np.mean(odd_depths) + 1e-9)
        else:
            feats["even_odd_ratio"] = 1.0
    else:
        feats["even_odd_ratio"] = 1.0

    # --- Centroid / flux symmetry (blend diagnostic) ---
    mid = len(flux) // 2
    feats["flux_symmetry"] = np.mean(flux[:mid]) - np.mean(flux[mid:])

    # --- High-frequency power (noise characterization) ---
    fft_vals = np.abs(np.fft.rfft(flux))
    freqs    = np.fft.rfftfreq(len(flux))
    feats["hf_power_ratio"] = np.sum(fft_vals[freqs > 0.1]) / (np.sum(fft_vals) + 1e-9)

    return feats


# ── STEP 4: Main Pipeline ─────────────────────────────────────────────────────
def run_pipeline():
    import lightkurve  # verify install

    # --- 1. Download ExoFOP catalog ---
    toi_df = fetch_exofop_catalog()

    # --- 2. Select TIC IDs per class ---
    label_df = select_tic_ids_from_toi(toi_df, n_per_class=N_PER_CLASS)

    if len(label_df) == 0:
        print("ERROR: No TIC IDs selected. Check catalog column names above.")
        return

    # --- 3. Fetch light curves and extract features ---
    print(f"\n[3/5] Fetching {len(label_df)} light curves from MAST (TESS)...")
    print("      This may take 10–30 minutes depending on your connection.\n")

    records = []
    total = len(label_df)

    for counter, (_, row) in enumerate(label_df.iterrows(), start=1):
        tic_id    = row["TIC_ID"]
        label     = row["label"]
        label_int = row["label_int"]

        print(f"   [{counter:3d}/{total}] TIC {tic_id:10d} | class: {label:8s}", end=" ... ", flush=True)

        lc_time, flux, meta = fetch_light_curve(tic_id)

        if lc_time is None:
            print("SKIPPED (no data)")
            continue

        # Resample to fixed length
        _, flux_resampled = resample_flux(lc_time, flux)

        # Extract features
        feats = extract_features(lc_time, flux)

        # Build record
        record = {
            "TIC_ID"       : tic_id,
            "label"        : label,
            "label_int"    : label_int,
            "n_points_raw" : meta.get("n_points_raw", 0),
        }
        record.update(feats)

        # Add raw resampled flux columns (FLUX_0 ... FLUX_999)
        for j, val in enumerate(flux_resampled):
            record[f"FLUX_{j}"] = round(val, 8)

        records.append(record)
        print(f"OK | SNR={feats['SNR']:.2f} | period={feats['ls_period']:.3f}d | depth={feats['transit_depth']:.6f}")

        time_module.sleep(0.3)   # polite rate limiting

    # --- 4. Build final DataFrame ---
    print(f"\n[4/5] Building final dataset from {len(records)} stars...")
    final_df = pd.DataFrame(records)

    # Reorder: metadata + features + raw flux columns
    meta_cols  = ["TIC_ID", "label", "label_int", "n_points_raw"]
    feat_cols  = [c for c in final_df.columns if c not in meta_cols and not c.startswith("FLUX_")]
    flux_cols  = [c for c in final_df.columns if c.startswith("FLUX_")]

    final_df = final_df[meta_cols + feat_cols + flux_cols]

    # --- 5. Save CSV ---
    print(f"\n[5/5] Saving to {OUTPUT_CSV} ...")
    final_df.to_csv(OUTPUT_CSV, index=False)

    size_mb = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"  DONE! Dataset saved: {OUTPUT_CSV}")
    print(f"  Rows    : {len(final_df)} stars")
    print(f"  Columns : {len(final_df.columns)}")
    print(f"  Size    : {size_mb:.1f} MB")
    print(f"{'='*60}")
    print("\nClass distribution:")
    print(final_df["label"].value_counts().to_string())
    print(f"\nKey feature columns:")
    for col in feat_cols:
        print(f"  {col}")
    print(f"\nRaw flux columns: FLUX_0 ... FLUX_{FLUX_POINTS-1}")

    return final_df


# ── STEP 5: Utility — Preview the Dataset ─────────────────────────────────────
def preview_dataset(csv_path=OUTPUT_CSV):
    """Call this after the pipeline to inspect the output."""
    df = pd.read_csv(csv_path)
    print(f"\nDataset shape: {df.shape}")
    print(df[["TIC_ID", "label", "SNR", "ls_period",
              "transit_depth", "transit_duration_days",
              "n_dips", "even_odd_ratio"]].head(20).to_string())


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = run_pipeline()
    if df is not None:
        preview_dataset()
        