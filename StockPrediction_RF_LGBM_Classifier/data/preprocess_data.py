import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import glob

def load_and_preprocess_data(csv_file_path):
    """
    加载单个CSV文件并完成预处理
    核心步骤：数据读取、日期处理、目标变量构造、特征筛选
    :param csv_file_path: CSV文件路径
    :return: 预处理后的df（完整数据）、X（特征矩阵）、y（目标变量）、dates（日期列）、feature_cols（特征列名）
    """
    # 第一步：读取CSV文件，捕获文件不存在的异常
    try:
        df = pd.read_csv(csv_file_path)
        print(f"成功读取数据文件：{csv_file_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"未找到数据文件，请检查路径：{csv_file_path}")

    # 日期处理：转换为datetime格式，按日期升序排列，删除缺失值
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)  # 保证数据按时间顺序
    df = df.dropna()  # 简单处理缺失值，实际业务中可根据情况补充

    # 构造目标变量：预测次日涨跌（1=涨，0=跌）
    df['next_day_return'] = df['daily_return'].shift(-1)  # 取次日的日收益率
    df['target'] = np.where(df['next_day_return'] > 0, 1, 0)  # 收益率>0为涨，否则为跌
    df = df.dropna(subset=['target'])  # 最后一行无次日数据，删除

    # 选择核心特征：基于前期业务分析筛选的有效特征
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 
                    'RSI_14', 'MACD', 'BB_width', 'volatility_20', 'daily_return']
    
    # 验证特征列是否存在：避免因数据格式问题导致后续训练报错
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据文件 {csv_file_path} 中缺失特征列：{missing_cols}")
    
    # 提取特征矩阵和目标变量，保留日期列用于后续滚动窗口划分
    X = df[feature_cols]
    y = df['target']
    dates = df['date']
    
    return df, X, y, dates, feature_cols

def get_all_csv_files(data_dir):
    """
    遍历指定目录下所有CSV文件
    :param data_dir: 数据目录路径
    :return: 所有CSV文件的绝对路径列表
    """
    # 先验证目录是否存在
    if not os.path.exists(data_dir):
        raise NotADirectoryError(f"数据目录不存在：{data_dir}")
    
    # 遍历目录下所有.csv文件（不区分大小写）
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"数据目录 {data_dir} 下未找到任何CSV文件")
    
    # 打印找到的文件数量和文件名，方便确认数据源
    print(f"共发现 {len(csv_files)} 个CSV文件：{[os.path.basename(f) for f in csv_files]}")
    return csv_files

# 单独运行该文件时的测试逻辑：验证数据加载和预处理流程
if __name__ == "__main__":
    # 拼接数据目录路径：当前文件在data目录，项目根目录是上级的上级
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(project_root, 'Stock_predict_analytics_csv')
    
    # 获取所有CSV文件，取第一个文件测试预处理流程
    csv_files = get_all_csv_files(data_dir)
    if csv_files:
        df, X, y, dates, feature_cols = load_and_preprocess_data(csv_files[0])