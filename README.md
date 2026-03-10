# AAPlot - Animal Behavior and Fiberphotometry Analysis Tools

AAPlot is a comprehensive suite of Python-based tools for analyzing animal behavior and analyzed Fiberphotometry data from Spike2. The toolkit includes modules for locomotion analysis and fluorescence trace analysis.

## Environment Setup

Requires Python 3.12 (3.13 for ARM64) and Anaconda in your PATH.  Create and
activate the `PLOT` environment with:

```bash
conda env create -n PLOT python=3.12.7 pandas numpy matplotlib seaborn ipykernel
conda activate PLOT
```

(Select `PLOT` as the kernel in VS Code when running notebooks.)

### Animal locomotion (EzTrack)

- **Script**: `Animal_locomotion_analysis_batch.py`

  Processes a directory of EzTrack CSV outputs (time, speed/position, optional
  markers).  Data are cleaned, smoothed, and aligned to an event time derived
  from markers or manually specified overrides.  The script computes speed
  summaries over predefined windows (pre‑1h, 0‑1h, 1‑2h, 2‑3h), saves a
  processed CSV and speed plot for each animal, and writes a combined summary
  CSV.

  Edit the `folder_path` and `manual_event_times` variables near the top of
  the file to configure defaults.

### Fiber photometry (Spike2 traces)

- **Script**: `fiber_trace_spike2_analysis_batch.py`

  Reads all Spike2-exported `.txt` files in a folder.  Each file must contain
  three columns: time (s), z‑score signal, and event marker (0/1).  Traces are
  aligned to the marker, smoothed, and metrics (peak mean, AUC, max/min) are
  calculated within configurable windows.  The script produces a metrics CSV
  and a `plot.svg` illustrating individual and mean traces.

  Configuration options such as axis limits, smoothing parameters, and
  analysis windows live at the top of the script or may be overridden via the
  command line.


## Usage

Run the following scripts for headless batch processing.

**Locomotion:**
```bash
python Animal_locomotion_analysis_batch.py  # edit folder_path/manual_event_times
```
Outputs live in `combined_analysis` (cleaned CSVs, plots, `batch_summary.csv`).

**Neural activity:**
```bash
python fiber_trace_spike2_analysis_batch.py <input_folder> [--output <folder>]
```

or with custom parameters:
```bash
python fiber_trace_spike2_analysis_batch.py 
```
(defaults and plotting options are in the script).  



* Created by Ruyi Cai @ Yulong Li lab, PKU, China

