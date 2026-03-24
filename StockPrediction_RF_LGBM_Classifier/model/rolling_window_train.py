import sys
import os
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

def get_window_splits(df, X, y):
    """
    生成滚动窗口划分，返回每个窗口的训练/验证/测试数据
    核心逻辑：以1天为步长滚动，每个窗口包含36个月训练数据+1个月验证数据+1天测试数据
    :param df: 预处理后的完整数据（需包含date列）
    :param X: 特征矩阵
    :param y: 目标变量
    :return: 生成器，逐个返回每个窗口的划分结果（包含各数据集和时间范围）
    """
    # 窗口参数配置：根据业务需求设定的时间范围
    initial_train_months = 36  # 初始训练窗口：取36个月的历史数据
    val_months_len = 1         # 验证窗口：取1个月的数据
    # 滚动步长：1天，测试窗口：1天（每次只预测次日涨跌）

    # 获取所有交易日（数据已按日期升序排列）
    all_dates = df['date'].values

    # 确定第一个可用于测试的日期：需要满足"36个月训练+1个月验证"的前置数据要求
    min_date = df['date'].min()
    first_test_date = min_date + relativedelta(months=initial_train_months + val_months_len)

    # 筛选出所有符合条件的测试日（测试日需晚于第一个可用测试日）
    test_dates = df.loc[df['date'] >= first_test_date, 'date'].unique()
    total_windows = len(test_dates)
    print(f"滚动窗口总数：{total_windows}个 ( 步长1天, 测试窗口1天 )")

    # 遍历每个测试日，生成对应的训练/验证/测试数据集
    for i, test_date in enumerate(test_dates):
        test_date = pd.Timestamp(test_date)

        # 确定验证期时间范围：测试日前1个月
        val_end = test_date - pd.Timedelta(days=1)
        val_start = test_date - relativedelta(months=val_months_len)

        # 确定训练期时间范围：验证期前的36个月
        train_end = val_start - pd.Timedelta(days=1)
        train_start = val_start - relativedelta(months=initial_train_months)

        # 按日期筛选对应的数据子集
        train_mask = (df['date'] >= train_start) & (df['date'] <= train_end)
        val_mask = (df['date'] >= val_start) & (df['date'] <= val_end)
        test_mask = df['date'] == test_date

        # 返回当前窗口的所有信息，包括窗口ID、时间范围、各数据集
        yield {
            'window_id': i,
            'test_date': test_date,
            'train_start': train_start,
            'train_end': train_end,
            'val_start': val_start,
            'val_end': val_end,
            'X_train': X[train_mask], 'y_train': y[train_mask],
            'X_val': X[val_mask], 'y_val': y[val_mask],
            'X_test': X[test_mask], 'y_test': y[test_mask],
        }

# 直接运行该文件时的测试逻辑：读取第一个CSV文件，遍历打印部分窗口信息
if __name__ == '__main__':
    # 把上级目录和项目根目录加入系统路径，方便导入自定义模块
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from data.preprocess_data import load_and_preprocess_data, get_all_csv_files
    
    # 拼接数据目录路径（项目根目录下的Stock_predict_analytics_csv）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(project_root, 'Stock_predict_analytics_csv')
    
    # 获取目录下所有CSV文件，取第一个做测试
    csv_files = get_all_csv_files(data_dir)
    if csv_files:
        df, X, y, dates, feature_cols = load_and_preprocess_data(csv_files[0])
        # 遍历滚动窗口，每50个窗口打印一次信息，避免输出过多
        for w in get_window_splits(df, X, y):
            i = w['window_id']
            if i % 50 == 0:
                print(f"窗口 {i+1}: 测试日 {w['test_date'].strftime('%Y-%m-%d')}")