import sys
import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import warnings
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# 日志重定向类：把控制台输出同时写入日志文件，方便后续排查问题
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout  # 保留原始控制台输出
        self.log = open(log_file, "a", encoding="utf-8")  # 追加模式写入日志，避免覆盖历史
        self.is_active = True  # 控制日志是否生效的开关

    def write(self, message):
        # 同时写入控制台和日志文件，写入后立即刷新，避免缓存导致日志丢失
        if self.is_active:
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
            os.fsync(self.log.fileno())  # 强制写入磁盘，确保日志不丢失

    def flush(self):
        # 兼容系统标准输出的flush方法，避免报错
        self.terminal.flush()
        self.log.flush()

    def close(self):
        # 关闭日志时先停用，再关闭文件
        self.is_active = False
        self.log.close()

# 初始化日志函数：创建日志目录，重定向stdout和stderr到日志文件
def init_logger(log_path):
    # 先创建日志所在目录（如果不存在）
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # 重定向标准输出和错误输出到自定义Logger
    sys.stdout = Logger(log_path)
    sys.stderr = sys.stdout  # 让错误信息也同步到日志里
    return sys.stdout

# 过滤无关警告：避免LGBM特征名相关的警告刷屏，不影响核心逻辑
warnings.filterwarnings(
    'ignore',
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names"
)

# 导入滚动窗口划分函数
from rolling_window_train import get_window_splits

def train_model_for_csv(csv_file_path, save_root_dir):
    """为单个CSV文件执行完整的模型训练流程
    包括：数据加载、滚动窗口遍历、模型调参、预测评分、结果保存
    :param csv_file_path: 单个CSV文件的路径
    :param save_root_dir: 模型和结果的根保存目录
    :return: 该CSV文件的所有训练结果
    """
    # 加载预处理数据：需要把上级目录加入路径才能导入data模块
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from data.preprocess_data import load_and_preprocess_data
    df, X, y, dates, feature_cols = load_and_preprocess_data(csv_file_path)
    
    # 创建该CSV专属的保存目录：用文件名命名，避免不同文件结果混叠
    csv_filename = os.path.basename(csv_file_path).replace('.csv', '')
    save_dir = os.path.join(save_root_dir, csv_filename)
    os.makedirs(save_dir, exist_ok=True)

    # 定义要训练的模型和超参数网格：根据业务经验选的常用参数范围
    models = {
        'RandomForest': {
            'estimator': RandomForestClassifier(random_state=42),  # 固定随机种子保证结果可复现
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
            },
        },
        'LightGBM': {
            'estimator': lgb.LGBMClassifier(random_state=42, verbose=-1),  # 关闭LGBM的冗余输出
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'num_leaves': [15, 31],
            },
        },
    }

    # 初始化变量：标准化器、结果存储字典、最后一轮模型存储
    scaler = StandardScaler()  # 特征标准化，提升模型收敛效果
    all_results = {name: [] for name in models}
    last_models = {}
    # 先计算总窗口数，方便进度条显示
    window_generator = get_window_splits(df, X, y)
    total_windows = sum(1 for _ in get_window_splits(df, X, y))
    print(f"\n开始训练 {csv_filename}（共{total_windows}个窗口）")
    print(f"模型：{', '.join(models.keys())}")
    window_generator = get_window_splits(df, X, y)  # 重置生成器（因为上面求和时遍历过一次）

    # 初始化进度条：优化显示样式，方便查看训练进度
    pbar = tqdm(
        window_generator, 
        total=total_windows, 
        desc=f"训练{csv_filename}",
        leave=True,  # 保留进度条历史，方便回看
        ncols=120,   # 加宽进度条，避免信息换行
        position=0,  # 固定进度条位置，不随输出跳动
        file=sys.stdout  # 强制输出到标准输出，保证和日志同步
    )

    # 遍历每个滚动窗口，逐窗口训练模型
    for w in pbar:
        i = w['window_id']
        test_date = w['test_date']
        X_train, y_train = w['X_train'], w['y_train']
        X_val, y_val = w['X_val'], w['y_val']
        X_test, y_test = w['X_test'], w['y_test']

        # 跳过空数据窗口：避免因数据不足导致训练报错
        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            pbar.set_description(f"训练{csv_filename} | 窗口{i+1} 数据为空，跳过")
            continue

        # 特征标准化：训练集fit_transform，验证/测试集仅transform避免数据泄露
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # 存储当前窗口各模型的评分
        window_scores = {}

        # 遍历每个模型，执行调参、训练、预测、评分
        for model_name, cfg in models.items():
            # 网格搜索调参：5折交叉验证，用准确率作为评分指标
            grid_search = GridSearchCV(
                cfg['estimator'],
                cfg['param_grid'],
                cv=5,
                scoring='accuracy',
                n_jobs=-1,  # 用所有CPU核心加速调参
            )
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_  # 取最优参数的模型

            # 保存当前窗口的最优模型：方便后续复现和预测
            model_path = os.path.join(save_dir, f'{model_name}_window_{i+1}.joblib')
            joblib.dump(best_model, model_path)

            # 预测：验证集全量预测，测试集（1天）取第一个结果
            val_preds = best_model.predict(X_val_scaled)
            test_pred = best_model.predict(X_test_scaled)[0] if len(X_test_scaled) > 0 else None
            test_actual = int(y_test.values[0]) if len(y_test) > 0 else None

            # 计算评分：验证集算准确率和F1，测试集只算单条结果的准确率
            val_acc = round(accuracy_score(y_val, val_preds), 4)
            val_f1 = round(f1_score(y_val, val_preds, average='binary'), 4)
            test_acc = round(1.0 if (test_pred == test_actual) else 0.0, 4) if test_pred is not None else None

            # 存储当前窗口的详细结果：方便后续分析每个窗口的表现
            all_results[model_name].append({
                'window': i + 1,
                'test_date': test_date.strftime('%Y-%m-%d'),
                'train_size': len(X_train),
                'best_params': grid_search.best_params_,
                'val_preds': val_preds.tolist(),
                'val_actual': y_val.values.tolist(),
                'test_pred': int(test_pred) if test_pred is not None else None,
                'test_actual': test_actual,
                'val_accuracy': val_acc,
                'val_f1': val_f1,
                'test_accuracy': test_acc
            })

            # 记录当前模型的评分，用于进度条和详情打印
            window_scores[model_name] = {
                'val_acc': val_acc,
                'val_f1': val_f1,
                'test_acc': test_acc
            }

            # 保存最后一轮的模型，用于后续批量预测
            last_models[model_name] = best_model

        # 更新进度条的基础描述：显示当前窗口和测试日，方便实时查看
        pbar.set_description(f"训练{csv_filename} | 测试日：{test_date.strftime('%Y-%m-%d')} | 窗口：{i+1}/{total_windows}")
        
        # 构造当前窗口的详情文本，手动打印（避免tqdm进度条干扰换行）
        detail_text = (
            f"\n{'='*60}\n"
            f"窗口 {i+1} 详情（测试日：{test_date.strftime('%Y-%m-%d')}）\n"
            f"{'='*60}\n"
            f"验证集准确率：\n"
            f"  - RandomForest：{window_scores['RandomForest']['val_acc']}\n"
            f"  - LightGBM：{window_scores['LightGBM']['val_acc']}\n"
            f"验证集F1分数：\n"
            f"  - RandomForest：{window_scores['RandomForest']['val_f1']}\n"
            f"  - LightGBM：{window_scores['LightGBM']['val_f1']}\n"
            f"{'='*60}\n"
        )
        
        # 打印详情文本并强制刷新，确保实时输出
        sys.stdout.write(detail_text)
        sys.stdout.flush()

        # 进度条尾部补充简短评分信息：快速查看当前窗口核心指标
        pbar.set_postfix({
            "RF_acc": window_scores['RandomForest']['val_acc'],
            "LGB_acc": window_scores['LightGBM']['val_acc']
        })

    # 关闭进度条，避免后续输出混乱
    pbar.close()

    # 保存最后一轮的标准化器和模型：用于后续增量训练或预测
    joblib.dump(scaler, os.path.join(save_dir, 'last_scaler.joblib'))
    for model_name, model in last_models.items():
        joblib.dump(model, os.path.join(save_dir, f'{model_name}_final.joblib'))

    # 保存所有窗口的结果到JSON文件：方便后续数据分析和可视化
    results_path = os.path.join(save_dir, 'all_results.json')
    json.dump(
        {'all_results': all_results, 'total_windows': total_windows, 'csv_file': csv_file_path},
        open(results_path, 'w'),
        ensure_ascii=False,
        indent=2
    )
    
    # 打印该CSV的训练完成信息和评分汇总，方便快速了解整体表现
    print(f"\n{csv_filename} 训练完成！")
    print(f"模型/结果保存至：{save_dir}")
    
    # 计算并打印所有窗口的平均评分：直观展示模型整体效果
    print(f"\n{csv_filename} 评分汇总（所有窗口平均）：")
    for model_name in models.keys():
        valid_results = [r for r in all_results[model_name] if r['val_accuracy'] is not None]
        if not valid_results:
            print(f"  {model_name}：无有效评分")
            continue
        avg_val_acc = np.mean([r['val_accuracy'] for r in valid_results])
        avg_val_f1 = np.mean([r['val_f1'] for r in valid_results])
        avg_test_acc = np.mean([r['test_accuracy'] for r in valid_results if r['test_accuracy'] is not None])
        print(f"  {model_name}：")
        print(f"    - 验证集平均准确率：{avg_val_acc:.4f}")
        print(f"    - 验证集平均F1：{avg_val_f1:.4f}")
        print(f"    - 测试集平均准确率：{avg_test_acc:.4f}")

    return all_results, save_dir

# 批量训练入口函数：遍历指定目录下所有CSV，逐个训练
def batch_train_all_csv(data_dir, save_root_dir):
    # 导入获取CSV文件列表的函数
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from data.preprocess_data import get_all_csv_files
    csv_files = get_all_csv_files(data_dir)
    # 遍历每个CSV文件，异常捕获：单个文件失败不影响其他文件
    for csv_file in csv_files:
        try:
            train_model_for_csv(csv_file, save_root_dir)
        except Exception as e:
            print(f"\n处理 {csv_file} 失败：{str(e)}")
            continue

# 主程序入口：单独运行该文件时，初始化日志并执行批量训练
if __name__ == '__main__':
    # 拼接路径：当前目录是model，项目根目录是上级的上级
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(project_root, 'Stock_predict_analytics_csv')
    save_root_dir = os.path.join(current_dir, 'saved_models_batch')
    
    # 初始化日志：日志文件保存在model目录下的log.txt
    log_file_path = os.path.join(current_dir, 'log.txt')
    logger = init_logger(log_file_path)
    
    try:
        # 打印训练开始的日志头，方便后续查看日志时定位
        print(f"\n{'='*80}")
        print(f"训练日志 - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        print(f"{'='*80}")
        print(f"AAL 股票涨跌预测系统 - 批量CSV训练模式")
        print(f"{'='*80}\n")
        
        # 验证数据目录并打印，确认训练数据源
        print(f"[Step 1] 验证数据目录...")
        print(f"数据目录：{data_dir}\n")
        
        # 开始批量训练
        print(f"[Step 2] 开始批量训练所有CSV文件...")
        batch_train_all_csv(data_dir, save_root_dir)
    finally:
        # 无论训练是否成功，最后都要恢复标准输出并关闭日志
        logger.close()
        sys.stdout = logger.terminal
        print("\n训练结束，日志已保存至：", log_file_path)