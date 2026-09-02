import os
import argparse
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import correlate
from tabulate import tabulate
import csv

VALID_EXTENSIONS = ('.wav', '.flac', '.mp3', '.m4a')


# -----------------------------------------------------------------------------
# Path Sanitization & Validation
# -----------------------------------------------------------------------------

def sanitize_path(path: str) -> str:
    """Removes outer quotes, extra spaces, and hidden drag-and-drop artifacts."""
    if not path:
        return ""
    return path.strip('"' + "'" + ' ')


def validate_file(path: str) -> bool:
    """Check if file exists and has a supported extension."""
    clean_path = sanitize_path(path)
    if not os.path.isfile(clean_path):
        print(f"[Error] File not found: {clean_path}")
        return False
    if not clean_path.lower().endswith(VALID_EXTENSIONS):
        print(f"[Error] Unsupported file format for: {clean_path}")
        return False
    return True


# -----------------------------------------------------------------------------
# 1. Input Processing & Normalization
# -----------------------------------------------------------------------------

def load_and_normalize(ref_path: str, test_path: str):
    ref_path = sanitize_path(ref_path)
    test_path = sanitize_path(test_path)

    ref_info = sf.info(ref_path)
    target_sr = ref_info.samplerate

    ref_audio, _ = librosa.load(ref_path, sr=target_sr, mono=False)
    test_audio, _ = librosa.load(test_path, sr=target_sr, mono=False)

    if ref_audio.ndim == 1:
        ref_audio = np.stack([ref_audio, ref_audio])
    if test_audio.ndim == 1:
        test_audio = np.stack([test_audio, test_audio])

    min_len = min(ref_audio.shape[1], test_audio.shape[1])
    ref_audio = ref_audio[:, :min_len]
    test_audio = test_audio[:, :min_len]

    return ref_audio, test_audio, target_sr


# -----------------------------------------------------------------------------
# 2. Temporal Alignment
# -----------------------------------------------------------------------------

def align_signals(ref_channel: np.ndarray, test_channel: np.ndarray, max_shift_samples: int = 44100):
    search_len = min(len(ref_channel), max_shift_samples)
    ref_slice = ref_channel[:search_len]
    test_slice = test_channel[:search_len]

    corr = correlate(ref_slice, test_slice, mode='full')
    delay = corr.argmax() - (len(test_slice) - 1)

    if delay > 0:
        test_aligned = np.pad(test_channel, (delay, 0))[:len(test_channel)]
    elif delay < 0:
        test_aligned = test_channel[-delay:]
        test_aligned = np.pad(test_aligned, (0, -delay))[:len(test_channel)]
    else:
        test_aligned = test_channel

    return test_aligned, delay


# -----------------------------------------------------------------------------
# 3. Mathematical SDR Engine
# -----------------------------------------------------------------------------

def compute_sdr(ref_seg: np.ndarray, test_seg: np.ndarray, eps: float = 1e-7) -> float:
    ref_energy = np.sum(ref_seg ** 2)

    if ref_energy < eps:
        return np.nan

    test_energy = np.sum(test_seg ** 2)
    if test_energy < eps:
        return -np.inf

    alpha = np.dot(ref_seg, test_seg) / test_energy
    noise = ref_seg - (alpha * test_seg)
    noise_energy = np.sum(noise ** 2)

    if noise_energy < eps:
        return 100.0

    sdr = 10 * np.log10(ref_energy / noise_energy)
    return float(sdr)


# -----------------------------------------------------------------------------
# 4. Chunked Evaluation System
# -----------------------------------------------------------------------------

def evaluate_track(ref_path: str, test_path: str, chunk_duration: float = 10.0, align: bool = True):
    ref, test, sr = load_and_normalize(ref_path, test_path)

    if align:
        _, delay = align_signals(ref[0], test[0], max_shift_samples=sr)
        if delay != 0:
            for ch in range(test.shape[0]):
                test[ch], _ = align_signals(ref[ch], test[ch], max_shift_samples=sr)

    chunk_samples = int(chunk_duration * sr)
    total_samples = ref.shape[1]

    chunk_results = []

    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        if (end - start) < (sr * 0.5):
            continue

        time_label = f"{start / sr:.1f}s - {end / sr:.1f}s"

        sdr_left = compute_sdr(ref[0, start:end], test[0, start:end])
        sdr_right = compute_sdr(ref[1, start:end], test[1, start:end])

        valid_scores = [s for s in (sdr_left, sdr_right) if not np.isnan(s)]
        sdr_overall = np.mean(valid_scores) if valid_scores else np.nan

        chunk_results.append({
            "chunk": time_label,
            "sdr_left": sdr_left,
            "sdr_right": sdr_right,
            "sdr_overall": sdr_overall
        })

    return chunk_results


# -----------------------------------------------------------------------------
# 5. Reporting & CLI Interface
# -----------------------------------------------------------------------------

def process_batch(ref_path: str, test_paths: list[str], chunk_sec: float, align: bool):
    summary_table = []

    for t_path in test_paths:
        clean_tpath = sanitize_path(t_path)
        if not validate_file(clean_tpath):
            continue

        results = evaluate_track(ref_path, clean_tpath, chunk_duration=chunk_sec, align=align)

        lefts = [r["sdr_left"] for r in results if not np.isnan(r["sdr_left"])]
        rights = [r["sdr_right"] for r in results if not np.isnan(r["sdr_right"])]
        overalls = [r["sdr_overall"] for r in results if not np.isnan(r["sdr_overall"])]

        summary_table.append({
            "File": os.path.basename(clean_tpath),
            "Mean Left (dB)": np.round(np.mean(lefts), 2) if lefts else "N/A",
            "Mean Right (dB)": np.round(np.mean(rights), 2) if rights else "N/A",
            "Overall SDR (dB)": np.round(np.mean(overalls), 2) if overalls else "N/A",
            "_raw_chunks": results,
            "_raw_overall": np.mean(overalls) if overalls else -np.inf
        })

    summary_table.sort(key=lambda x: x["_raw_overall"], reverse=True)
    return summary_table


def export_csv(summary_data: list[dict], output_csv_path: str):
    clean_csv_path = sanitize_path(output_csv_path)
    if not clean_csv_path:
        return

    # Ensure .csv extension
    if not clean_csv_path.lower().endswith('.csv'):
        clean_csv_path += '.csv'

    with open(clean_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Summary Header
        writer.writerow(["File", "Mean Left (dB)", "Mean Right (dB)", "Overall SDR (dB)"])

        # Write only the aggregated metrics
        for item in summary_data:
            writer.writerow([
                item["File"],
                item["Mean Left (dB)"],
                item["Mean Right (dB)"],
                item["Overall SDR (dB)"]
            ])

    print(f"\nSummary report successfully exported to: {clean_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate audio stem quality against ground truth reference using SDR.")
    parser.add_argument("--ground-truth", "-g", type=str, help="Path to reference audio file")
    parser.add_argument("--ai", "-a", type=str, nargs="+", help="Path to one or more AI stem audio files or directory")
    parser.add_argument("--chunk-size", type=float, default=10.0,
                        help="Window evaluation chunk size in seconds (default: 10.0)")
    parser.add_argument("--no-align", action="store_true", help="Disable automatic temporal alignment")
    parser.add_argument("--csv", type=str, help="Path to export CSV report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-chunk breakdown to console")

    args = parser.parse_args()

    # --- Interactive Inputs with Path Sanitization ---
    ref_input = args.ground_truth or input("Enter ground-truth reference file path: ")
    ref_path = sanitize_path(ref_input)

    if not validate_file(ref_path):
        return

    if not args.ai:
        ai_input = input("Enter AI stem file or folder path: ")
        test_inputs = [ai_input]
    else:
        test_inputs = args.ai

    # Ask for CSV export if in interactive mode and argument wasn't passed
    csv_target = args.csv
    if args.csv is None and not args.ground_truth and not args.ai:
        user_csv = input("Enter CSV export path (press Enter to skip): ")
        csv_target = sanitize_path(user_csv)

    # Expand directories and sanitize all paths
    expanded_test_paths = []
    for path in test_inputs:
        clean_p = sanitize_path(path)
        if os.path.isdir(clean_p):
            for root, _, files in os.walk(clean_p):
                for f in files:
                    if f.lower().endswith(VALID_EXTENSIONS):
                        expanded_test_paths.append(os.path.join(root, f))
        else:
            expanded_test_paths.append(clean_p)

    if not expanded_test_paths:
        print("[Error] No valid audio files found to test.")
        return

    print("\nProcessing evaluations...")
    results = process_batch(ref_path, expanded_test_paths, chunk_sec=args.chunk_size, align=not args.no_align)

    if args.verbose:
        for item in results:
            print(f"\n--- Detailed Breakdown: {item['File']} ---")
            chunk_table = [[c["chunk"], c["sdr_left"], c["sdr_right"], c["sdr_overall"]] for c in item["_raw_chunks"]]
            print(tabulate(chunk_table, headers=["Chunk Window", "Left (dB)", "Right (dB)", "Overall (dB)"],
                           floatfmt=".2f"))

    print("\n=== SDR Evaluation Summary ===")
    display_summary = [
        {k: v for k, v in row.items() if not k.startswith("_")}
        for row in results
    ]
    print(tabulate(display_summary, headers="keys", tablefmt="fancy_grid"))

    # Export if a path was provided and not left blank
    if csv_target:
        export_csv(results, csv_target)


if __name__ == "__main__":
    main()