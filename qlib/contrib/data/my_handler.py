from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import Processor

class MyAlpha158(Alpha158):
    """继承 Alpha158，但覆盖因子配置"""
    
    def __init__(self, **kwargs):
        # 移除可能传入的 factors 参数
        kwargs.pop('factors', None)
        kwargs.pop('custom_factors', None)
        super().__init__(**kwargs)
    
    def get_feature_config(self):
        """覆盖因子配置，返回自定义因子"""
        return [
            # 动量因子
            '$close / Ref($close, 5) - 1',
            '$close / Ref($close, 10) - 1',
            '$close / Ref($close, 20) - 1',
            '$close / Ref($close, 60) - 1',

            # 均线偏离
            "Mean($close, 5) / $close",
            "Mean($close, 10) / $close",
            "Mean($close, 20) / $close",
            
            # 波动率
            "Std($close, 5)",
            "Std($close, 20)",
            
            # 价量相关性
            "Corr($close, $volume, 20)",
            
            # 价格形态
            "($high - $low) / $close",
            "($close - $open) / $open",
            
            # 成交量变化
            "Sum($volume, 5) / Sum($volume, 20)",
            "($volume - Mean($volume, 20)) / Mean($volume, 20)",
            
            # 排序
            #"Rank($close, 20)",
            
            # 长期收益率
            "($close - Ref($close, 20)) / Ref($close, 20)",
        ]
    
    def get_label_config(self):
        """覆盖标签配置"""
        return ["Ref($close, -5) / Ref($close, -1) - 1"], ["LABEL0"]
