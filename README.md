# AAPlot - Animal Behavior and Neural Activity Analysis Tools

AAPlot is a comprehensive suite of Python-based tools for analyzing animal behavior and neural activity data from Spike2 recordings. The toolkit includes modules for locomotion analysis, behavioral event detection, and fluorescence trace analysis.

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

### Animal Locomotion Analysis

1. **AAPlot_Animal_Locomotion_analysis[batch].ipynb**
   - Analyzes multiple animal locomotion data files in batch
   - Features:
     - Time-based analysis (pre-event and post-event)
     - Speed and distance calculations
     - Cumulative analysis windows (0-1h, 0-2h, 0-3h)
     - Exports combined summary and individual data files

2. **AAPlot_Animal_Locomotion_analysis[single].ipynb**
   - Single-file version of the locomotion analysis
   - Ideal for individual file processing and analysis verification

### Animal Behavior Analysis

1. **AAPlot_Animal_Behavior[PreTest].ipynb**
   - Extracts time and speed data from Spike2 files
   - Exports data to CSV format for further analysis

2. **AAPlot_Animal_Behavior[Develop].ipynb**
   - Development version with enhanced features
   - Aims to integrate data extraction, event detection, and evaluation
   - Currently under development

### Neural Activity Analysis

1. **AAPlot_spike2_trace[single]_v1.ipynb**
   - Processes single neural recording traces
   - Analyzes fluorescence data from Spike2 recordings

2. **AAPlot_spike2_trace[multi]_v1.ipynb**
   - Handles multiple neural recording traces
   - Batch processing capabilities for fluorescence data

3. **AAPlot_spike2_trace_optimized.ipynb**
   - Optimized version with improved performance
   - Enhanced data processing capabilities

4. **AAPlot_Event_dFF0.py**
   - Event-related fluorescence analysis
   - Calculates dF/F0 for neural activity quantification

### Optimized Tools (AAPLOT_optimized/)

The `AAPLOT_optimized` folder contains production-ready tools with enhanced performance:

- **Core Analysis Scripts:**
  - `optimized_aplot_notebook.py`: Main analysis notebook
  - `optimized_fluorescence_analyzer.py`: Fluorescence data processing
  - `run_fluorescence_analysis.py`: Standalone analysis script

- **Batch Processing:**
  - `run_multi_file_analysis.py`: Multiple file processing
  - `run_with_config.py`: Configuration-based analysis

- **Configuration:**
  - `config.yaml`: Analysis parameters and settings
  - `requirements.txt`: Required Python packages

- **Documentation:**
  - `README_优化说明.md`: Optimization details
  - `快速开始指南.md`: Quick start guide

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
- 🔄 Advanced Behavior Analysis (In Development)
- ✅ Neural Activity Analysis (Stable)
- ✅ Optimized Tools (Production Ready)

## Contributing

Feel free to submit issues and enhancement requests.