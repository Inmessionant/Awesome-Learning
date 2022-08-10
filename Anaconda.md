

------



```
# conda 创建环境
conda create -n name python=3.8

# 使用pytorch官网的pip/conda命令装torch和torchvision
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116


# MacOS安装
# MPS acceleration is available on MacOS 12.3+
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```




```
# 清理缓存
conda clean -a

# 安装requirements里面的版本
conda install --yes --file requirements.txt

# 测试cuda是否可用
import torch
import torchvision
print(torch.cuda.is_available())
print(torch.version.cuda)

# 删除conda环境
conda remove -n name --all

# conda换源记得去掉default，添加pytorch
```


