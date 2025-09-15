"""
荧光信号分析运行脚本
替换原始的 AAPlot_Event_dFF0.py 功能
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from optimized_fluorescence_analyzer import FluorescenceAnalyzer, AnalysisConfig

def main():
    """运行荧光信号分析"""
    print("=== AAPlot 荧光信号分析 ===")
    
    # 步骤1：配置参数
    config = AnalysisConfig()
    
    # 设置输出目录
    config.output_dir = './fluorescence_output'
    
    # 可选：调整分析参数
    config.pre_event_time = 20.0    # 事件前20秒
    config.post_event_time = 120.0  # 事件后120秒
    config.baseline_start_time = 100.0  # 基线计算开始时间
    config.baseline_end_time = 500.0    # 基线计算结束时间
    config.save_individual_traces = True  # 保存单独的轨迹文件
    config.save_plots = True             # 保存图表
    
    print(f"输出目录: {config.output_dir}")
    
    # 步骤2：指定输入文件
    # 请修改为您的实际CSV文件路径
    input_file = "example_data.csv"  # 替换为实际文件路径
    
    if not os.path.exists(input_file):
        print(f"❌ 错误: 输入文件不存在 - {input_file}")
        print("请修改 input_file 变量为实际的CSV数据文件路径")
        print("\n📝 文件格式要求:")
        print("   - CSV格式文件")
        print("   - 三列数据: 时间, 信号值, 事件标记")
        print("   - 事件标记列中，事件发生时值为1，其他为0")
        return
    
    try:
        # 步骤3：初始化分析器
        analyzer = FluorescenceAnalyzer(config)
        
        # 步骤4：运行分析
        print(f"🔬 正在分析文件: {input_file}")
        if analyzer.run_analysis(input_file):
            print("✅ 荧光信号分析完成")
            
            # 显示结果摘要
            print("\n" + "="*50)
            print("📋 分析结果摘要")
            print("="*50)
            
            print(f"📊 处理轨迹数量: {len(analyzer.baseline_subtracted_traces)}")
            print(f"📈 时间轴长度: {len(analyzer.time_axis) if analyzer.time_axis is not None else 0} 点")
            print(f"⏰ 时间范围: {analyzer.time_axis[0]:.2f} - {analyzer.time_axis[-1]:.2f} 秒" if analyzer.time_axis is not None else "N/A")
            
            print(f"\n📁 输出文件:")
            output_files = [
                "merged_dff0_traces.csv",
                "merged_zscore_traces.csv", 
                "fluorescent_dF/F0_plot.png",
                "fluorescent_z-score_plot.png"
            ]
            
            for file_name in output_files:
                file_path = os.path.join(config.output_dir, file_name)
                if os.path.exists(file_path):
                    print(f"   ✅ {file_name}")
                else:
                    print(f"   ❌ {file_name} (未生成)")
            
            print(f"\n✅ 分析完成！结果保存在: {config.output_dir}")
            
        else:
            print("❌ 荧光信号分析失败")
            
    except Exception as e:
        print(f"❌ 执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def show_file_format_example():
    """显示文件格式示例"""
    print("\n📝 CSV文件格式示例:")
    print("Time,dF_F0,Event")
    print("0.0,0.1,0")
    print("0.1,0.12,0")
    print("0.2,0.15,0")
    print("...")
    print("100.0,0.2,1")  # 事件标记
    print("100.1,0.25,0")
    print("...")

if __name__ == "__main__":
    # 显示文件格式要求
    show_file_format_example()
    
    # 运行分析
    main()
