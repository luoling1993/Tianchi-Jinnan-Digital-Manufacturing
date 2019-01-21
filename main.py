#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from automl import auto_ml
from blending import blending_main
from processing import processing_main
from tree_models import model_main

if __name__ == '__main__':
    t0 = time.time()

    # 特征工程，若以生成，可注释
    processing_main()

    # 树模型
    """
    @model: lgb, xgb 或者stacking
    @selector: 是否使用筛选特征
    @force: 结合selector使用
    """
    model_main(model='stacking', selector=True, force=False)

    # automl 模型
    """
    @selector: 是否使用赛选特征
    """
    auto_ml(selector=False)

    # 模型简单融合
    """
    @automl_rate: automl分数权重
    """
    blending_main(automl_rate=0.6)

    print(f'usage time: {time.time() - t0} seconds')
