@echo off
REM AAPlot 命令行使用示例

echo === AAPlot 命令行使用示例 ===

REM 激活conda环境（如果使用conda）
REM conda activate aplot_optimized

echo.
echo 1. 基本使用 - 分析单个CSV文件:
echo python optimized_fluorescence_analyzer.py your_data.csv

echo.
echo 2. 自定义参数 - 调整时间窗口:
echo python optimized_fluorescence_analyzer.py your_data.csv --pre-event 30 --post-event 150

echo.
echo 3. 指定输出目录:
echo python optimized_fluorescence_analyzer.py your_data.csv --output-dir ./my_results

echo.
echo 4. 保存单独轨迹文件:
echo python optimized_fluorescence_analyzer.py your_data.csv --save-individual

echo.
echo 5. 不生成图表:
echo python optimized_fluorescence_analyzer.py your_data.csv --no-plots

echo.
echo 6. 完整参数示例:
echo python optimized_fluorescence_analyzer.py your_data.csv ^
echo   --pre-event 30 ^
echo   --post-event 150 ^
echo   --baseline-start 100 ^
echo   --baseline-end 500 ^
echo   --output-dir ./results ^
echo   --save-individual

echo.
echo 请将 "your_data.csv" 替换为您的实际数据文件路径
pause
