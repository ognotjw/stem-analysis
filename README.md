# Audio SDR Evaluator

A Python utility for measuring how accurately AI-separated audio tracks (stems) match a ground-truth reference using Signal-to-Distortion Ratio (SDR). Supports batch folder processing, format normalization, temporal alignment, and chunked evaluation.

## Prerequisites

Ensure you have Python 3.9+ installed, along with the required dependencies:

```bash
pip install numpy librosa soundfile scipy tabulate museval
```

## Setup & Usage

### Interactive Mode
Run the script without flags to be prompted for inputs:

```bash
python eval.py
```

### Command-Line Arguments

| Flag | Short | Description | Default |
| :--- | :--- | :--- | :--- |
| `--ground-truth` | `-g` | Path to reference audio file | *Interactive Prompt* |
| `--ai` | `-a` | Path to AI stem file(s) or directory | *Interactive Prompt* |
| `--chunk-size` | | Window evaluation chunk size in seconds | `10.0` |
| `--no-align` | | Disable cross-correlation time alignment | `Enabled` |
| `--museval` | | Use Museval BSSEval engine (SDR, ISR, SAR) | `Disabled`|
| `--verbose` | `-v` | Display detailed logs | `Disabled` |
| `--csv` | | Path to export CSV summary report | *Optional* |

## Examples

**Evaluate a single stem:**
```bash
python eval.py -g "ref.wav" -a "stem.wav"
```

**Batch evaluate a folder of stems with custom chunk size and CSV export:**
```bash
python eval.py -g "ref.flac" -a "./stems_folder/" --chunk-size 5.0 --csv "report.csv"
```

## Additional Notes

* **Supported Formats:** `.wav`, `.flac`, `.mp3`, `.m4a`.
* **Path Handling:** Input paths (including those from terminal drag-and-drop) are automatically sanitized to remove surrounding quotes and extra spaces.
* **Channel Standardization:** Single-channel (mono) audio files are automatically converted to dual-channel (stereo) prior to processing.

**This project was made using AI.**
