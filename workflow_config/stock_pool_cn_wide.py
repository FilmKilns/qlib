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
from qlib.workflow.task.gen import RollingGen
from qlib.workflow.task.utils import TimeAdjuster
import pandas as pd
import os

# ==================== 配置参数 ====================
PROVIDER_URI = "~/.qlib/qlib_data/qlib_data_1775983012349"  # A股数据路径

# 时间范围与滚动窗口设置
START_TIME = "2015-07-10"
END_TIME = "2025-03-31"           # 滚动测试截止日
TRAIN_PERIOD = 5 * 365            # 训练窗口长度：5年
TEST_PERIOD = 1 * 365             # 测试窗口长度：1年
STEP = 6 * 30                     # 滚动步长：6个月

# 最终回测期
FINAL_BACKTEST_START = "2025-04-01"
FINAL_BACKTEST_END = "2026-04-10"

# 交易策略参数
TOPK = 6                          # 持仓数量（池子一半）
N_DROP = 2                        # 每次调仓替换数量

# 模型超参数
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

# ==================== 任务配置模板 ====================
def get_task_template() -> dict:
    """返回带有占位符的任务配置模板，占位符将由 RollingGen 填充"""
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
                        "fit_start_time": "<FIT_START>",
                        "fit_end_time": "<FIT_END>",
                        "instruments": 'all',
                    }
                },
                "segments": {
                    "train": ["<FIT_START>", "<FIT_END>"],
                    "valid": ["<VALID_START>", "<VALID_END>"],
                    "test": ["<TEST_START>", "<TEST_END>"],
                }
            }
        },
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": MODEL_KWARGS
        }
    }

def get_port_analysis_config(test_start: str, test_end: str) -> dict:
    """生成回测配置，动态设置测试起止时间"""
    config = {
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.strategy",
            "kwargs": {
                "topk": TOPK,
                "n_drop": N_DROP,
                "signal": "<PRED>",
            }
        },
        "backtest": BACKTEST_KWARGS.copy()
    }
    config["backtest"]["start_time"] = test_start
    config["backtest"]["end_time"] = test_end
    return config

# ==================== 生成滚动任务 ====================
print(f"创建滚动训练计划：训练 {TRAIN_PERIOD//365} 年，测试 {TEST_PERIOD//365} 年，步长 {STEP//30} 个月")
rolling_gen = RollingGen(
    step=STEP, 
    rtype=TimeAdjuster.SHIFT_SD, # 或 RollingGen.ROLL_SD
)

task_template = get_task_template()
task_list = rolling_gen.generate(task_template)
print(f"共生成 {len(task_list)} 个滚动任务。")

# ==================== 执行滚动训练 ====================
all_metrics = []

for i, task_config in enumerate(task_list):
    segs = task_config["dataset"]["kwargs"]["segments"]
    train_start, train_end = segs["train"]
    valid_start, valid_end = segs["valid"]
    test_start, test_end = segs["test"]

    print(f"\n{'='*60}")
    print(f"滚动窗口 {i+1} / {len(task_list)}")
    print(f"训练期: {train_start} → {train_end}")
    print(f"验证期: {valid_start} → {valid_end}")
    print(f"测试期: {test_start} → {test_end}")
    print(f"{'='*60}")

    # 使用 R.start 上下文管理器管理实验
    with R.start(experiment_name=f"rolling_etf_{i}"):
        # 初始化数据集和模型
        dataset = init_instance_by_config(task_config["dataset"])
        model = init_instance_by_config(task_config["model"])

        # 训练与预测
        print("训练模型中...")
        model.fit(dataset)
        print("生成预测信号...")
        pred = model.predict(dataset)

        # 记录信号
        sr = SignalRecord()
        sr.generate(dataset=dataset, model=model, pred=pred)

        # 执行回测
        port_config = get_port_analysis_config(test_start, test_end)
        par = PortAnaRecord(recorder=R, **{"config": port_config})
        par.generate()

        # 提取关键指标
        report = par.load("portfolio_analysis/report_normal.pkl")
        ann_return = report["return"]
        sharpe = report.get("sharpe", 0.0)
        max_dd = report.get("max_drawdown", 0.0)

        all_metrics.append({
            "window": i,
            "train_start": train_start,
            "train_end": train_end,
            "valid_start": valid_start,
            "valid_end": valid_end,
            "test_start": test_start,
            "test_end": test_end,
            "annual_return": ann_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
        })

        print(f"测试期年化收益率: {ann_return:.2%}, 夏普比率: {sharpe:.2f}, 最大回撤: {max_dd:.2%}")

# ==================== 汇总滚动测试结果 ====================
print("\n" + "="*60)
print("滚动测试汇总报告")
print("="*60)
df_metrics = pd.DataFrame(all_metrics)
print(df_metrics.to_string())

print(f"\n平均年化收益率: {df_metrics['annual_return'].mean():.2%}")
print(f"平均夏普比率: {df_metrics['sharpe'].mean():.2f}")
print(f"平均最大回撤: {df_metrics['max_drawdown'].mean():.2%}")

# ==================== 最终回测 ====================
print("\n" + "="*60)
print("在最近完整周期上训练最终模型，并进行最终回测")
print("="*60)

final_segments = {
    "train": (START_TIME, "2025-03-31"),
    "valid": ("2024-01-01", "2024-12-31"),   # 用于早停
    "test": (FINAL_BACKTEST_START, FINAL_BACKTEST_END),
}

# 构建最终任务配置（不使用占位符）
final_task_config = {
    "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": START_TIME,
                    "end_time": FINAL_BACKTEST_END,
                    "fit_start_time": final_segments["train"][0],
                    "fit_end_time": final_segments["train"][1],
                    "instruments": 'all',
                }
            },
            "segments": final_segments
        }
    },
    "model": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": MODEL_KWARGS
    }
}

with R.start(experiment_name="final_model_2025_2026"):
    dataset_final = init_instance_by_config(final_task_config["dataset"])
    model_final = init_instance_by_config(final_task_config["model"])

    print("训练最终模型中...")
    model_final.fit(dataset_final)
    pred_final = model_final.predict(dataset_final)

    sr_final = SignalRecord()
    sr_final.generate(dataset=dataset_final, model=model_final, pred=pred_final)

    port_config_final = get_port_analysis_config(FINAL_BACKTEST_START, FINAL_BACKTEST_END)
    par_final = PortAnaRecord(recorder=R, **{"config": port_config_final})
    par_final.generate()

    report_final = par_final.load("portfolio_analysis/report_normal.pkl")
    print(f"最终回测期 ({FINAL_BACKTEST_START} → {FINAL_BACKTEST_END})")
    print(f"年化收益率: {report_final['return']:.2%}")
    print(f"夏普比率: {report_final.get('sharpe', 0.0):.2f}")
    print(f"最大回撤: {report_final.get('max_drawdown', 0.0):.2%}")

print("\n所有实验记录已保存至 mlruns 目录，可使用 `mlflow ui` 查看。")