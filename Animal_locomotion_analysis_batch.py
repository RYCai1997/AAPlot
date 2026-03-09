# Animal Locomotion - Speed Analysis & Batch Statistics

# **Functionality:**
# 1. Input: a folder from EzTrack output `.csv` files
# 2. for each file: read, clean, compute speed, smooth and plot, save outputs
# 3. aggregate summary statistics across files (mean speeds, distances, etc.)
# 4. export both individual cleaned data and combined summary CSV

# Run in `PLOT` environment
    
# The `PLOT` enviorment ?
# - Python3.12.7
# - pandas
# - numpy
# - matplotlib
# - seaborn
# - ipykernel

# *Warning*

# *! Make sure you have installed `Anaconda`，and added to PATH *
# *! Make sure you have already confiured the `PLOT` environment, 
# *!    If not, run this command: `conda env create -n PLOT python=3.12.7 pandas numpy matplotlib seaborn ipykernel` 
# *!    If you are using ARM64 CPU, use Python3.13.3 intead，using `conda-forge` instead of `conda`)*
# *Apply `PLOT` in VScode ：Select in the Kernel*
# *Apply `PLOT` in terminal：`conda activate PLOT`*
# ------------------------------------- By Ruyi Cai @ Yulong Li Lab, PKU, China ----------------------------------

# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import glob
import warnings

# pandas occasionally emits a FutureWarning when modifying a slice; we
# already handle data in-place explicitly so it's safe to silence.
warnings.filterwarnings('ignore', category=FutureWarning)


# -------------------------- parameter input --------------------------
# configuration block – edit these values before running the script

# 1. path to the directory containing EzTrack CSV output files.  The script
#    will scan this folder recursively for files that end in `.csv`.
folder_path = r'D:\Temp\DrugIntake behavior Analysis\Cocaine\TEST'

# 2. optional manual event times keyed by file prefix (characters before the
#    first underscore).  When specified, the given value (seconds) will be
#    used instead of trying to infer an event from a `Marker` column. (if existed) 
#    the default event time marker is at 3600s, if no input or marker is found 
#    leave entries blank for files where the marker is present.
manual_event_times = {
    'C89': 2750,
    'C90': 2590,
    'C91': 3000,
    'C92': 3600,
    'C93': 3700,
    'C94': 3800,
    'C95': 3600,
    'C96': 3600
}


# ----------------------Functions ---------------------------
# read files under the folder
def read_data_file(filepath):
    """Simply load the CSV and return the DataFrame."""
    return pd.read_csv(filepath)


def process_file(filepath, output_dir, event_time_override=None, verbose=True):
    """Clean, analyze and plot a single file; return summary dict.

    Parameters
    ----------
    filepath : str
        Path to input file.
    output_dir : str
        Directory where plots and cleaned CSVs will be written.
    event_time_override : float or None
        If provided, this value (seconds) will be used as the event time
        instead of trying to infer it from a marker column. Useful when the
        file contains no markers.
    """
    if verbose:
        print(f"\n--- processing {filepath} ---")
    df = read_data_file(filepath)
    base = os.path.splitext(os.path.basename(filepath))[0]
    if verbose:
        print(f"--- base name {base} ---")
        print(f"columns: {list(df.columns)}")

    # convert eztrack style if present
    if 'Frame' in df.columns and 'Distance_cm' in df.columns:
        time_secs = df['Frame'] / 30
        speed = df['Distance_cm'] * 30
        dfc = pd.DataFrame({'Time': time_secs, 'Speed': speed})
        dfc['Marker'] = 0
    else:
        dfc = df.copy()
        if 'Time' not in dfc.columns or 'Speed' not in dfc.columns:
            raise ValueError('input file missing Time/Speed columns')
        dfc['Time'] = pd.to_numeric(dfc['Time'], errors='coerce')
        dfc['Speed'] = pd.to_numeric(dfc['Speed'], errors='coerce')
        if 'Marker' not in dfc.columns:
            dfc['Marker'] = 0

    # data cleaning, delete impossible values and replace the NAN
    dfc['Speed'] = dfc['Speed'].clip(upper=100)
    dfc['Speed'] = dfc['Speed'].interpolate()

    # determine event time
    if event_time_override is not None: # user defined
        event_time = event_time_override
        print(f"using manual event time {event_time} s")
    else:
        ev = dfc[dfc['Marker'] == 1] # If there is a marker column, named 'Marker', 1 for event, 0 otherwise
        if len(ev):
            event_time = ev.iloc[0]['Time']
            print(f"inferred event time {event_time:.2f} s from marker")
        else:
            event_time = 3600  # default to 1h session if no marker or override
            print(f"no marker; using midpoint event time {event_time:.2f} s")
    dfc['Time_hours'] = (dfc['Time'] - event_time) / 3600

    # stats 
    windows = {
        'pre_1h': (-1, 0),
        'post_0_to_1h': (0, 1),
        'post_1_to_2h': (1, 2),
        'post_2_to_3h': (2, 3),
    }
    speed_stats = {}
    dist_stats = {}
    for name, (start, end) in windows.items():
        mask = (dfc['Time_hours'] >= start) & (dfc['Time_hours'] < end)
        w = dfc[mask]
        if len(w):
            speed_stats[name] = {'mean': w['Speed'].mean(),
                                  'std': w['Speed'].std(),
                                  'n': len(w)}
            speeds = w['Speed'] * 3600
            dt = np.diff(w['Time_hours'])
            dist = np.sum(speeds[:-1] * dt)
            dist_stats[name] = dist / 100
        else:
            speed_stats[name] = {'mean': np.nan, 'std': np.nan, 'n': 0}
            dist_stats[name] = np.nan

    # plot
    smooth = savgol_filter(dfc['Speed'], window_length=min(31,len(dfc['Speed']))//2*2+1, polyorder=2)
    smooth = np.clip(smooth,0,None)
    fig, ax = plt.subplots(figsize=(6,2.5))
    ax.plot(dfc['Time'], smooth, color='k', linewidth=1)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Speed (cm/s)')
    ax.set_title(base)
    plt.tight_layout()
    plotfile = os.path.join(output_dir, f"{base}_speed.svg")
    fig.savefig(plotfile, dpi=300)
    plt.close(fig)

    # save cleaned data
    datafile = os.path.join(output_dir, f"{base}_processed.csv")
    dfc.to_csv(datafile, index=False)

    # build summary dictionary and return
    summary = {'file': base}
    summary.update({f"speed_{k}": v['mean'] for k, v in speed_stats.items()})
    summary.update({f"dist_{k}": v for k, v in dist_stats.items()})

    return summary

# ---------------------- workflow description ------------------------
# The flow of the script is as follows:
#
#   1. gather a list of ``.csv`` files from ``folder_path``
#   2. create ``combined_analysis`` subdirectory for output
#   3. loop over each input file:
#        a. compute the prefix and look up any manual event time override
#        b. call ``process_file`` (silent by default) which:
#             • loads/cleans the data
#             • infers or uses the provided event time
#             • computes summary statistics and distance
#             • smooths & plots speed and saves the figure & cleaned CSV
#             • returns a dictionary of summary values
#        c. append result dict to ``all_results`` list
#   4. if any results were collected, convert ``all_results`` to a DataFrame
#      and write a batch summary CSV
#
# This section implements steps 1–4; you normally do _not_ need to change it.
#
# main batch processing code
# ---------------------------------------------------------------
# collect files
data_files = []
for ext in ['.csv']:
    data_files.extend(glob.glob(os.path.join(folder_path, f'*{ext}')))

if not data_files:
    raise ValueError(f"No .csv files found in {folder_path}")

print(f"Found {len(data_files)} files")
for f in data_files:
    print("-", os.path.basename(f))


# output directory for all results
output_dir = os.path.join(folder_path, 'combined_analysis')
os.makedirs(output_dir, exist_ok=True)

all_results = []
# dictionary mapping file-prefix (chars before first underscore) -> event time (seconds)


for fp in data_files:
    base = os.path.splitext(os.path.basename(fp))[0]
    prefix = base.split('_', 1)[0]
    override = manual_event_times.get(prefix)
    if override is not None:
        print(f"using manual override for prefix {prefix}: {override}")
    try:
        res = process_file(fp, output_dir, event_time_override=override, verbose=False)
        all_results.append(res)
    except Exception as e:
        print(f"error processing {os.path.basename(fp)}: {e}")

# produce dataframe and save summary
if all_results:
    summary_df = pd.DataFrame(all_results)
    summary_file = os.path.join(output_dir, 'batch_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print("Batch summary saved to", summary_file)
    print(summary_df.head())
else:
    print("No results produced.")
