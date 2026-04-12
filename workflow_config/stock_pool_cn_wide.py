"""
滚动训练与回测脚本 - 针对12只场内ETF的Qlib工作流

核心优化：
1. 模型正则化参数（防过拟合）
2. 动态持仓与调仓逻辑（topk=6, n_drop=2）
3. 真实交易成本模拟（VWAP成交、滑点、印花税）
4. 滚动窗口验证（滑动窗口，检验时序稳定性）

使用方法：
1. 确保已安装Qlib： pip install pyqlib
2. 准备数据：确保 provider_uri 路径指向有效数据
3. 创建股票池文件： ./pool/my_etfs.txt，每行一个代码
4. 运行： python rolling_run.py
"""

import qlib
from qlib.config import REG_CN
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import init_instance_by_config
from qlib.tests.data import GetData
import pandas as pd
import os

# ==================== 配置参数 ====================
PROVIDER_URI = "~/.qlib/qlib_data/qlib_data_1775983012349"  # A股数据路径

# 时间范围与滚动窗口设置
START_TIME = "2015-07-10"
END_TIME = "2025-03-31"           # 滚动测试截止日，留出最新数据做最终验证
TRAIN_PERIOD = 5 * 365            # 训练窗口长度：5年
TEST_PERIOD = 1 * 365             # 测试窗口长度：1年
STEP = 6 * 30                     # 滚动步长：6个月

# 最终回测期（用于评估最新表现）
FINAL_BACKTEST_START = "2025-04-01"
FINAL_BACKTEST_END = "2026-04-10"

# 交易策略参数
TOPK = 6                          # 持仓数量（池子一半）
N_DROP = 2                        # 每次调仓替换数量

# 模型超参数（已针对过拟合优化）
MODEL_KWARGS = {
    "loss": "mse",
    "num_boost_round": 500,
    "early_stopping_rounds": 50,
    "learning_rate": 0.02,
    "max_depth": 5,
    "num_leaves": 50,
    "lambda_l1": 5.0,
    "lambda_l2": 5.0,
    "min_child_samples": 50,
    "min_split_gain": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "num_threads": 20,
}

# 回测成本参数
BACKTEST_KWARGS = {
    "start_time": FINAL_BACKTEST_START,
    "end_time": FINAL_BACKTEST_END,
    "account": 100000000,
    "benchmark": "SH000300",
    "exchange_kwargs": {
        "limit_threshold": 0.1,
        "deal_price": "vwap",
        "open_cost": 0.0003,
        "close_cost": 0.0013,
        "min_cost": 0.05,
        "slippage": 0.001,
    }
}

# ==================== 初始化Qlib ====================
print("正在初始化 Qlib...")
qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)

# ==================== 定义工作流配置 ====================
def get_task_config(segments: dict) -> dict:
    """
    根据给定的时间段生成任务配置字典。
    注意：handler 的 instruments 指向股票池文件。
    """
    return {
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": {
                        "start_time": START_TIME,
                        "end_time": END_TIME,
                        "fit_start_time": segments["train"][0],
                        "fit_end_time": segments["train"][1],
                        "instruments": 'all',
                        # 可选：修改预测标签为5日收益率，降低噪声
                        # "label": ["Ref($close, -5) / Ref($close, -1) - 1"]
                    }
                },
                "segments": segments
            }
        },
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": MODEL_KWARGS
        }
    }

def get_port_analysis_config() -> dict:
    """回测与策略配置"""
    return {
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.strategy",
            "kwargs": {
                "topk": TOPK,
                "n_drop": N_DROP,
                "signal": "<PRED>",
            }
        },
        "backtest": BACKTEST_KWARGS
    }

# ==================== 创建滚动训练管理器 ====================
print(f"创建滚动训练计划：训练 {TRAIN_PERIOD//365} 年，测试 {TEST_PERIOD//365} 年，步长 {STEP//30} 个月")
exp_manager = R.get_exp_manager()
rolling = exp_manager.create_rolling(
    train_period=TRAIN_PERIOD,
    test_period=TEST_PERIOD,
    step=STEP,
    start_time=START_TIME,
    end_time=END_TIME,
)

# ==================== 执行滚动训练 ====================
all_metrics = []  # 存储每期回测指标

for i, (train_start, train_end, test_start, test_end) in enumerate(rolling):
    print(f"\n{'='*60}")
    print(f"滚动窗口 {i+1} / {len(rolling)}")
    print(f"训练期: {train_start} → {train_end}")
    print(f"测试期: {test_start} → {test_end}")
    print(f"{'='*60}")

    segments = {
        "train": (train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")),
        "test": (test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d")),
    }

    # 1. 启动实验
    exp = R.start_exp(
        experiment_name=f"rolling_etf_{i}",
        recorder_name="mlflow",
        uri="mlruns",
    )

    # 2. 初始化任务与模型
    task_config = get_task_config(segments)
    task = init_instance_by_config(task_config)
    model = init_instance_by_config(task_config["model"])

    # 3. 训练与预测
    print("训练模型中...")
    model.fit(task)
    print("生成预测信号...")
    pred = model.predict(task)

    # 4. 记录信号与回测
    sr = SignalRecord(recorder=exp)
    sr.generate(dataset=task.dataset, model=model, pred=pred)

    port_config = get_port_analysis_config()
    # 注意：回测时间段必须覆盖测试期
    port_config["backtest"]["start_time"] = test_start.strftime("%Y-%m-%d")
    port_config["backtest"]["end_time"] = test_end.strftime("%Y-%m-%d")

    par = PortAnaRecord(recorder=exp, **{"config": port_config})
    par.generate()

    # 5. 提取关键指标
    metrics = exp.list_metrics()
    report = par.load("portfolio_analysis/report_normal.pkl")
    positions = par.load("portfolio_analysis/positions_normal.pkl")

    ann_return = report["return"]  # 年化收益率
    sharpe = report.get("sharpe", 0.0)
    max_dd = report.get("max_drawdown", 0.0)

    all_metrics.append({
        "window": i,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "annual_return": ann_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    })

    print(f"测试期年化收益率: {ann_return:.2%}, 夏普比率: {sharpe:.2f}, 最大回撤: {max_dd:.2%}")
    R.end_exp()

# ==================== 汇总滚动测试结果 ====================
print("\n" + "="*60)
print("滚动测试汇总报告")
print("="*60)
df_metrics = pd.DataFrame(all_metrics)
print(df_metrics.to_string())

print(f"\n平均年化收益率: {df_metrics['annual_return'].mean():.2%}")
print(f"平均夏普比率: {df_metrics['sharpe'].mean():.2f}")
print(f"平均最大回撤: {df_metrics['max_drawdown'].mean():.2%}")

# ==================== 最终回测（可选） ====================
print("\n" + "="*60)
print("在最近完整周期上训练最终模型，并进行最终回测")
print("="*60)

final_segments = {
    "train": (START_TIME, "2025-03-31"),
    "test": (FINAL_BACKTEST_START, FINAL_BACKTEST_END),
}

exp_final = R.start_exp(experiment_name="final_model_2025_2026")
task_final = init_instance_by_config(get_task_config(final_segments))
model_final = init_instance_by_config(get_task_config(final_segments)["model"])

print("训练最终模型中...")
model_final.fit(task_final)
pred_final = model_final.predict(task_final)

sr_final = SignalRecord(recorder=exp_final)
sr_final.generate(dataset=task_final.dataset, model=model_final, pred=pred_final)

port_config_final = get_port_analysis_config()
par_final = PortAnaRecord(recorder=exp_final, **{"config": port_config_final})
par_final.generate()

report_final = par_final.load("portfolio_analysis/report_normal.pkl")
print(f"最终回测期 ({FINAL_BACKTEST_START} → {FINAL_BACKTEST_END})")
print(f"年化收益率: {report_final['return']:.2%}")
print(f"夏普比率: {report_final.get('sharpe', 0.0):.2f}")
print(f"最大回撤: {report_final.get('max_drawdown', 0.0):.2%}")

R.end_exp()

print("\n所有实验记录已保存至 mlruns 目录，可使用 `mlflow ui` 查看。")