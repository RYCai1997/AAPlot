# AAPlot - Animal Behavior and Fiberphotometry Analysis Tools

AAPlot is a comprehensive suite of Python-based tools for analyzing animal behavior and Fiberphotometry data from Spike2 recordings. The toolkit includes modules for locomotion analysis, behavioral event analysis, and fluorescence trace analysis.

## Environment Setup

### Prerequisites
- Python 3.12.7 (or Python 3.13.3 for ARM64 CPU)
- Anaconda (must be added to PATH)

### Creating the Environment
```bash
conda env create -n PLOT python=3.12.7 pandas numpy matplotlib seaborn ipykernel
# For ARM64 CPU, add conda-forge at the end of the command
```

### Activation
- In VS Code: Select 'PLOT' from the Kernel list
- In terminal: `conda activate PLOT`

## Tools Overview

### Animal Behavior Analysis Workflow

1. **AAPlot_Animal_Behavior Tools**
   - Purpose: Extract locomotion data and create event markers
   - Workflow:
     1. Use EzTrack to track animal movement
     2. Process EzTrack output data using AAPlot_Animal_Behavior
     3. Import processed data into Spike2 to specify event markers
     4. Export event-marked data as .txt file from Spike2

2. **AAPlot_Animal_Locomotion_analysis Tools**
   - Purpose: Analyze behavior with event markers
   - Input: Spike2 exported .txt files with event markers
   - Features:
     - **[batch].ipynb**: Process multiple files with:
       - Time-based analysis (pre-event and post-event)
       - Speed and distance calculations
       - Cumulative analysis windows (0-1h, 0-2h, 0-3h)
       - Combined summary export
     - **[single].ipynb**: Single-file processing for verification

### Fiber Photometry Analysis

1. **AAPlot_spike2_trace[multi]_v1.ipynb**
   - Purpose: Analyze fiber photometry fluorescent signals
   - Features:
     - Process multiple recording traces
     - Analyze fluorescence data from Spike2 recordings
     - Batch processing capabilities

2. **AAPlot_spike2_trace[single]_v1.ipynb**
   - Single trace version of fluorescence analysis
   - Detailed analysis of individual recordings

3. **AAPlot_spike2_trace_optimized.ipynb**
   - Optimized version with improved performance
   - Enhanced data processing capabilities

4. **AAPlot_Event_dFF0.py**
   - Event-related fluorescence analysis
   - Calculates dF/F0 for neural activity quantification

### Optimized Tools (AAPLOT_optimized/)


## Usage

### For Locomotion Analysis
1. Open `AAPlot_Animal_Locomotion_analysis[batch].ipynb` for multiple files or `[single].ipynb` for individual files
2. Set the input folder path (for batch) or file path (for single)
3. Run all cells
4. Results will be saved in an 'analysis_results' subfolder

### For Neural Activity Analysis
1. Choose the appropriate notebook based on your needs (single/multi/optimized)
2. Configure the input parameters
3. Run the analysis
4. Check the output folder for results

### For Optimized Tools
1. Navigate to the `AAPLOT_optimized` folder
2. Configure parameters in `config.yaml`
3. Run the appropriate script:
   - `run_command_line.bat` for Windows users
   - `run_fluorescence_analysis.py` for direct Python execution
   - `run_multi_file_analysis.py` for batch processing

## Development Status

- ✅ Locomotion Analysis (Stable)
- ✅ Basic Behavior Analysis (Stable)
- ✅ Neural Activity Analysis (Stable)
- ✅ Optimized Tools (Production Ready)

## Contributing

Feel free to submit issues and enhancement requests.