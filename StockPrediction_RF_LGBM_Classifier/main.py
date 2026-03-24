"""
股票涨跌预测 - 批量训练主入口（全局模型增量训练）
=====================================
执行流程：
1. 遍历 Stock_predict_analytics_csv 目录下所有CSV文件
2. 用全局共享模型增量训练每个CSV的滚动窗口数据
3. 最终生成1个全局模型 + 每个CSV的独立结果
"""

import sys
import os
import time

# 定义项目路径相关常量
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
DATA_DIR = os.path.join(PROJECT_ROOT, '..', 'Stock_predict_analytics_csv')
SAVE_ROOT_DIR = os.path.join(MODEL_DIR, 'saved_models_batch')

# 将模型目录和项目根目录加入系统路径，方便模块导入
sys.path.insert(0, MODEL_DIR)
sys.path.insert(0, PROJECT_ROOT)

if __name__ == '__main__':
    print("=" * 80)
    print("股票涨跌预测系统 - 全局模型增量训练模式")
    print("=" * 80)

    try:
        # 第一步：验证数据目录是否存在，保证后续训练有数据来源
        print(f"\n[Step 1] 验证数据目录...")
        if not os.path.exists(DATA_DIR):
            raise NotADirectoryError(f"数据目录不存在：{DATA_DIR}")
        print(f"数据目录：{DATA_DIR}")

        # 第二步：开始增量训练，记录训练总耗时
        print("\n[Step 2] 开始增量训练所有CSV文件（共享全局模型）...")
        start_time = time.time()
        
        # 导入批量训练函数并执行
        from model.train_and_tune import batch_train_all_csv
        batch_train_all_csv(DATA_DIR, SAVE_ROOT_DIR)
        
        # 计算并打印总耗时
        elapsed = time.time() - start_time
        print(f"\n所有CSV文件增量训练完成！总耗时：{elapsed:.1f} 秒")

        # 打印模型和结果的保存路径，方便后续查找
        print("\n" + "=" * 80)
        print(f"全局模型保存至：{os.path.join(SAVE_ROOT_DIR, 'global_model')}")
        print(f"各CSV结果保存至：{SAVE_ROOT_DIR}（按CSV文件名分目录）")
        print("=" * 80)

    # 捕获并打印执行过程中的异常，异常时退出程序
    except Exception as e:
        print(f"\n程序执行出错：{str(e)}")
        sys.exit(1)