import argparse
import csv
import os
import time
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import correlate
from tabulate import tabulate

try:
    import museval
except ImportError:
    museval = None

VALID_EXTENSIONS = ('.wav', '.flac', '.mp3', '.m4a')


# -----------------------------------------------------------------------------
# Helpers & I/O
# -----------------------------------------------------------------------------

def sanitize_path(path: str) -> str:
    """Removes quotes, whitespace, and drag-and-drop artifacts."""
    return path.strip('"' + "'" + ' ') if path else ""


def validate_file(path: str) -> bool:
    """Validates existence and format of audio files."""
    clean_path = sanitize_path(path)
    if not os.path.isfile(clean_path):
        print(f"[Error] File not found: {clean_path}")
        return False
    if not clean_path.lower().endswith(VALID_EXTENSIONS):
        print(f"[Error] Unsupported format: {clean_path}")
        return False
    return True


def load_audio(path: str, target_sr: int = None, verbose: bool = False):
    """Loads audio file and standardizes shape to stereo (2, samples)."""
    clean_path = sanitize_path(path)
    if verbose:
        print(f"   [IO] Reading audio info for: {os.path.basename(clean_path)}")
        
    if target_sr is None:
        target_sr = sf.info(clean_path).samplerate

    if verbose:
        print(f"   [IO] Loading waveform (Sample Rate: {target_sr} Hz)...")
        
    start_time = time.time()
    audio, sr = librosa.load(clean_path, sr=target_sr, mono=False)
    load_duration = time.time() - start_time

    if audio.ndim == 1:
        if verbose:
            print("   [IO] Mono audio detected. Converting to pseudo-stereo.")
        audio = np.stack([audio, audio])

    if verbose:
        print(f"   [IO] Audio loaded in {load_duration:.2f}s | Shape: {audio.shape} ({audio.shape[1] / sr:.2f} seconds)")

    return audio, sr


def align_signals(ref_channel: np.ndarray, test_channel: np.ndarray, max_shift_samples: int = 44100, verbose: bool = False):
    """Aligns test channel signal with reference channel based on cross-correlation."""
    if verbose:
        print("   [Align] Computing cross-correlation for temporal alignment...")
        
    start_time = time.time()
    search_len = min(len(ref_channel), max_shift_samples)
    corr = correlate(ref_channel[:search_len], test_channel[:search_len], mode='full')
    delay = corr.argmax() - (search_len - 1)
    align_duration = time.time() - start_time

    if verbose:
        delay_ms = (delay / max_shift_samples) * 1000
        print(f"   [Align] Alignment calculated in {align_duration:.2f}s | Shift delay: {delay} samples ({delay_ms:.2f} ms)")

    if delay > 0:
        return np.pad(test_channel, (delay, 0))[:len(test_channel)], delay
    elif delay < 0:
        aligned = np.pad(test_channel[-delay:], (0, -delay))[:len(test_channel)]
        return aligned, delay
    
    return test_channel, 0


# -----------------------------------------------------------------------------
# Evaluation Engines
# -----------------------------------------------------------------------------

def compute_sdr(ref_seg: np.ndarray, test_seg: np.ndarray, eps: float = 1e-7) -> float:
    """Computes mathematical SDR for single-channel signal segments."""
    ref_energy = np.sum(ref_seg ** 2)
    if ref_energy < eps:
        return np.nan

    test_energy = np.sum(test_seg ** 2)
    if test_energy < eps:
        return -np.inf

    alpha = np.dot(ref_seg, test_seg) / test_energy
    noise = ref_seg - (alpha * test_seg)
    noise_energy = np.sum(noise ** 2)

    return 100.0 if noise_energy < eps else float(10 * np.log10(ref_energy / noise_energy))


def compute_museval_fast(ref_audio: np.ndarray, test_audio: np.ndarray, sr: int, win_sec: float = 1.0, hop_sec: float = 1.0, verbose: bool = False) -> dict:
    """Evaluates full track using correctly-scaled sample parameters in Museval."""
    if museval is None:
        raise RuntimeError("Museval is not installed. Run: pip install museval")

    # Shape conversion: (channels, samples) -> (sources, samples, channels)
    references = ref_audio.T[np.newaxis, :, :]
    estimates = test_audio.T[np.newaxis, :, :]

    # Convert window and hop from SECONDS to SAMPLES
    win_samples = int(win_sec * sr)
    hop_samples = int(hop_sec * sr)

    if verbose:
        print(f"   [Museval] Initializing BSS Eval engine...")
        print(f"   [Museval] Frame config -> Window: {win_sec}s ({win_samples} samples) | Hop: {hop_sec}s ({hop_samples} samples)")
        print(f"   [Museval] Input matrix shape: {references.shape}")
        print(f"   [Museval] Running evaluation algorithms (SDR, ISR, SAR)...")

    start_time = time.time()

    try:
        sdr, isr, _, sar = museval.evaluate(
            references,
            estimates,
            win=win_samples,
            hop=hop_samples,
            padding=True
        )

        eval_duration = time.time() - start_time
        if verbose:
            print(f"   [Museval] Computation completed in {eval_duration:.2f} seconds.")

        def avg(arr):
            v = np.asarray(arr, dtype=float)
            v = v[np.isfinite(v)]
            return float(np.mean(v)) if v.size else np.nan

        metrics = {"sdr": avg(sdr[0]), "isr": avg(isr[0]), "sar": avg(sar[0])}

        if verbose:
            print(f"   [Museval] Computed metrics: SDR={metrics['sdr']:.2f} dB | ISR={metrics['isr']:.2f} dB | SAR={metrics['sar']:.2f} dB")

        return metrics

    except Exception as exc:
        print(f"[Warning] Museval evaluation failed: {exc}")
        return {"sdr": np.nan, "isr": np.nan, "sar": np.nan}


# -----------------------------------------------------------------------------
# Core Batch Pipeline
# -----------------------------------------------------------------------------

def evaluate_track(ref_audio: np.ndarray, test_path: str, sr: int, chunk_sec: float = 10.0, align: bool = True, use_museval: bool = False, verbose: bool = False):
    """Processes evaluation for a single track pair."""
    file_name = os.path.basename(test_path)
    if verbose:
        print(f"\n--- Processing Track: {file_name} ---")

    test_audio, _ = load_audio(test_path, target_sr=sr, verbose=verbose)
    
    # Match audio track lengths
    min_len = min(ref_audio.shape[1], test_audio.shape[1])
    if verbose:
        print(f"   [Prep] Trimming tracks to matching length: {min_len} samples ({min_len / sr:.2f}s)")
        
    ref = ref_audio[:, :min_len]
    test = test_audio[:, :min_len]

    if align:
        for ch in range(test.shape[0]):
            if verbose:
                print(f"   [Prep] Channel {ch + 1} Alignment:")
            test[ch], _ = align_signals(ref[ch], test[ch], max_shift_samples=sr, verbose=verbose)

    if use_museval:
        scores = compute_museval_fast(ref, test, sr, verbose=verbose)
        return [{
            "chunk": "Full Track",
            "sdr_left": scores["sdr"],
            "sdr_right": scores["sdr"],
            "sdr_overall": scores["sdr"],
            "isr": scores["isr"],
            "sar": scores["sar"]
        }]

    # Custom Segmented SDR Pipeline
    if verbose:
        print(f"   [SDR] Executing custom segmented SDR across {chunk_sec}s chunks...")
        
    chunk_samples = int(chunk_sec * sr)
    chunk_results = []
    total_chunks = len(range(0, min_len, chunk_samples))

    for idx, start in enumerate(range(0, min_len, chunk_samples), 1):
        end = min(start + chunk_samples, min_len)
        if (end - start) < (sr * 0.5):
            continue

        sdr_l = compute_sdr(ref[0, start:end], test[0, start:end])
        sdr_r = compute_sdr(ref[1, start:end], test[1, start:end])
        valid = [s for s in (sdr_l, sdr_r) if not np.isnan(s)]
        overall = np.mean(valid) if valid else np.nan

        time_label = f"{start / sr:.1f}s - {end / sr:.1f}s"
        if verbose:
            print(f"   [SDR Chunk {idx}/{total_chunks}] {time_label} -> Left: {sdr_l:.2f} dB | Right: {sdr_r:.2f} dB | Avg: {overall:.2f} dB")

        chunk_results.append({
            "chunk": time_label,
            "sdr_left": sdr_l,
            "sdr_right": sdr_r,
            "sdr_overall": overall,
            "isr": np.nan, "sar": np.nan
        })

    return chunk_results


def process_batch(ref_path: str, test_paths: list[str], chunk_sec: float, align: bool, use_museval: bool = False, verbose: bool = False):
    """Executes evaluation over all valid test targets."""
    print("\n========================================================")
    print("              LOADING REFERENCE AUDIO                   ")
    print("========================================================")
    ref_audio, sr = load_audio(ref_path, verbose=verbose)
    summary_table = []

    print("\n========================================================")
    print("             STARTING BATCH EVALUATION                  ")
    print("========================================================")
    
    total_files = len(test_paths)
    batch_start_time = time.time()

    for idx, t_path in enumerate(test_paths, 1):
        clean_path = sanitize_path(t_path)
        if not validate_file(clean_path):
            continue

        print(f"\n[Batch Progress: File {idx}/{total_files}]")
        results = evaluate_track(ref_audio, clean_path, sr, chunk_sec, align, use_museval, verbose=verbose)

        def safe_mean(key):
            vals = [r[key] for r in results if np.isfinite(r.get(key, np.nan))]
            return np.round(np.mean(vals), 2) if vals else "N/A"

        overalls = [r["sdr_overall"] for r in results if np.isfinite(r["sdr_overall"])]

        summary_table.append({
            "File": os.path.basename(clean_path),
            "Metric": "Museval BSSEval" if use_museval else "Custom SDR",
            "Mean Left (dB)": safe_mean("sdr_left"),
            "Mean Right (dB)": safe_mean("sdr_right"),
            "Overall SDR (dB)": safe_mean("sdr_overall"),
            "ISR (dB)": safe_mean("isr"),
            "SAR (dB)": safe_mean("sar"),
            "_raw_chunks": results,
            "_raw_overall": np.mean(overalls) if overalls else -np.inf
        })

    total_batch_duration = time.time() - batch_start_time
    print(f"\n[Batch Complete] Evaluated {len(summary_table)} files in {total_batch_duration:.2f} seconds.")

    summary_table.sort(key=lambda x: x["_raw_overall"], reverse=True)
    return summary_table


# -----------------------------------------------------------------------------
# Reporting & CLI Entry
# -----------------------------------------------------------------------------

def export_csv(summary_data: list[dict], output_csv_path: str, verbose: bool = False):
    """Exports structured results to CSV file."""
    clean_path = sanitize_path(output_csv_path)
    if not clean_path.endswith(".csv"):
        clean_path += ".csv"

    if verbose:
        print(f"\n[Export] Writing evaluation summary to: {clean_path}")

    headers = ["File", "Metric", "Mean Left (dB)", "Mean Right (dB)", "Overall SDR (dB)", "ISR (dB)", "SAR (dB)"]

    with open(clean_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for item in summary_data:
            writer.writerow([item[h] for h in headers])

    print(f"Summary successfully exported to: {clean_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate audio stem quality against reference audio.")
    parser.add_argument("--ground-truth", "-g", type=str, help="Path to reference audio file")
    parser.add_argument("--ai", "-a", type=str, nargs="+", help="Path to AI audio files or directory")
    parser.add_argument("--chunk-size", type=float, default=10.0, help="Chunk window in seconds (default: 10.0)")
    parser.add_argument("--no-align", action="store_true", help="Disable automatic temporal alignment")
    parser.add_argument("--csv", type=str, help="Path to export CSV report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-chunk breakdown and verbose step execution logs")
    parser.add_argument("--museval", action="store_true", help="Use Museval BSSEval metrics")
    args = parser.parse_args()

    ref_path = sanitize_path(args.ground_truth or input("Enter ground-truth file path: "))
    if not validate_file(ref_path):
        return

    test_inputs = args.ai if args.ai else [input("Enter AI stem file or folder path: ")]

    expanded_paths = []
    for path in test_inputs:
        clean_p = sanitize_path(path)
        if os.path.isdir(clean_p):
            for root, _, files in os.walk(clean_p):
                expanded_paths.extend([os.path.join(root, f) for f in files if f.lower().endswith(VALID_EXTENSIONS)])
        else:
            expanded_paths.append(clean_p)

    if not expanded_paths:
        print("[Error] No valid audio files found to test.")
        return

    print(f"\nProcessing evaluations using {'Museval BSSEval' if args.museval else 'Custom SDR'}...")
    results = process_batch(ref_path, expanded_paths, args.chunk_size, not args.no_align, args.museval, verbose=args.verbose)

    if args.verbose:
        for item in results:
            print(f"\n--- Detailed Breakdown: {item['File']} ---")
            chunk_table = [[c["chunk"], c["sdr_left"], c["sdr_right"], c["sdr_overall"]] for c in item["_raw_chunks"]]
            print(tabulate(chunk_table, headers=["Chunk Window", "Left (dB)", "Right (dB)", "Overall (dB)"], floatfmt=".2f"))

    print("\n=== SDR Evaluation Summary ===")
    display_summary = [{k: v for k, v in row.items() if not k.startswith("_")} for row in results]
    print(tabulate(display_summary, headers="keys", tablefmt="fancy_grid"))

    if args.csv:
        export_csv(results, args.csv, verbose=args.verbose)


if __name__ == "__main__":
    main()