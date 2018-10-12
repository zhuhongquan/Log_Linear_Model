# Log_Linear_Model
## 一、目录文件
    ./data/:
        train.conll: 训练集
        dev.conll: 开发集
    ./big_data/:
        train.conll: 训练集
        dev.conll: 开发集
        test.conll: 测试集
    ./:
        log-linear-model.py: 代码(未进行特征抽取优化)
        log-linear-model-partial-feature.py: 代码(特征抽取优化)
    ./README.md: 使用说明

## 二、运行
### 1.运行环境
    python 3
### 2.运行方法
    各个参数
    'train_data_file': 'data/train.conll', #训练集文件,大数据改为'../big_data/train.conll'
    'dev_data_file': 'data/dev.conll',     #开发集文件,大数据改为'../big_data/dev.conll'
    'test_data_file': 'data/dev.conll',    #测试集文件,大数据改为'../big_data/test.conll'
    'iterator': 100,                          # 最大迭代次数
    'batchsize': 50,                          # 批次大小
    'shuffle': False,                         # 每次迭代是否打乱数据
    'exitor': 10,                             # 连续多少个迭代没有提升就退出
    'regulization': False,                    # 是否正则化
    'step_opt': False,                        # 是否步长优化
    'eta': 0.5,                               # 初始步长
    'C': 0.0001                               # 正则化系数,regulization为False时无效
    
### 3.参考结果
#### (1)小数据测试
