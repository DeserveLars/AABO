# AABO
在MMDetection框架下实现AABO
# 1、AABO自适应锚框优化方法的介绍
ECCV 2020 论文代码：AABO: 通过贝叶斯子采样进行对象检测的自适应锚框优化`（[论文](https://arxiv.org/abs/2007.09336)）`。<br>在 AABO 中，该论文提出了一种通过贝叶斯子采样进行对象检测的自适应锚框优化方法，其中自动确定特定数据集和检测器的最佳锚框配置，无需手动调整。<br>实验证明了 AABO 在不同检测器和数据集上的有效性，例如在 COCO 上实现了约 2.4% 的 mAP 改进，并且最佳锚点仅通过优化锚点配置就可以在 SOTA 检测器上带来 1.4% 到 2.4% 的 mAP 改进，例如将 Mask RCNN 从 40.3% 提升到 42.3%，将 HTC 检测器从 46.8% 提升到 48.2%。
# 2、数据集介绍
## 2.1 coco2017
COCO是一个常见的对象检测数据集，包含80个对象类，包含118K训练图像(train)、5K验证图像(val)和20K个未注释测试图像(test-dev)。<br>可以在官网下载数据集，以下是下载并解压数据集的命令：<br>
```python
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
```
得到如下目录：<br>
```python
/path/to/coco/
    ├── annotations/
    │   ├── instances_train2017.json
    │   ├── instances_val2017.json
    ├── train2017/
    │   ├── 000000000001.jpg
    │   ├── ...
    ├── val2017/
    │   ├── 000000000001.jpg
    │   ├── ...
```
## 2.2 VG(Visual Genome)和ADE
VG 和 ADE 是两个具有数千个对象类的大规模对象检测基准。<br>确保已经下载了 VG 和 ADE20K 数据集，并将它们放在合适的位置。得到如下目录：<br>
```python
/path/to/vg/
    ├── annotations/
    │   ├── instances_train.json
    │   ├── instances_val.json
    ├── train/
    │   ├── 000000000001.jpg
    │   ├── ...
    ├── val/
    │   ├── 000000000001.jpg
    │   ├── ...

/path/to/ade/
    ├── annotations/
    │   ├── instances_train.json
    │   ├── instances_val.json
    ├── train/
    │   ├── ADE_train_00000001.jpg
    │   ├── ...
    ├── val/
    │   ├── ADE_val_00000001.jpg
    │   ├── ...
```
# 3、具体执行步骤
## 3.1 安装依赖
```python
#创建虚拟环境
conda create -n mmdetection python=3.8

#安装torch和torchvision
mkdir deps
cd deps
wget -c https://download.pytorch.org/whl/cu111/torch-1.8.0%2Bcu111-cp38-cp38-linux_x86_64.whl
wget -c https://download.pytorch.org/whl/cu111/torchvision-0.9.0%2Bcu111-cp38-cp38-linux_x86_64.whl

#安装mmdetection
pip install mmcv-full
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .

#请用对应的aabo文件替代mmdetection里对应的文件：
- `AABO/__init__.py`--->`mmdetection/mmdet/models/anchor_heads/__init__.py`
- `AABO/anchor_generator.py`--->`mmdetection/mmdet/core/anchor/anchor_generator.py`
- `AABO/anchor_head.py`--->`mmdetection/mmdet/models/anchor_heads/anchor_head.py`
#请将这些文件添加到相应的目录中：
- 添加`AABO/aabo_rpn_head.py``mmdetection/mmdet/models/anchor_heads/`
- 添加`AABO/aabo_mask_rcnn_r101_fpn_2x.py ``mmdetection/configs/`
- 添加`AABO/aabo_htc_dcov_x101_64x4d_fpn_24e.py``mmdetection/configs/`
#注意有两个示例配置文件：`aabo_mask_rcnn_r101_fpn_2x.py`和`aabo_htc_dcov_x101_64x4d_fpn_24e.py`。使用这两个配置文件，AABO 搜索到的优化锚点设置可以提升 Mask RCNN 和 HTC 的性能。
如果您想在其他检测器上测试这些优化的锚点设置的性能，只需将默认锚点替换为这两个文件中记录的优化锚点即可。在论文中，对不同的高级基于锚点的检测器进行了实验，并观察到一致的性能改进。

#复制对应配置文件
mkdir -p configs/custom
cp configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py configs/custom/my_faster_rcnn_r50_fpn_1x_coco.py

#修改数据集对应的配置文件，例如：configs/dataset/coco.py
dataset_type = 'CocoDataset'
data_root = '/path/to/coco/'
```
# 4、代码运行步骤
## 4.1 训练数据集
使用以下命令开始训练模型：<br>
`python tools/train.py configs/custom/faster_rcnn_r50_fpn_1x_vg.py`
## 4.2 验证数据集
训练完成后，可以使用以下命令评估模型：<br>
`python tools/test.py configs/custom/faster_rcnn_r50_fpn_1x_vg.py work_dirs/faster_rcnn_r50_fpn_1x_vg/latest.pth --eval bbox`
# 5、实验结果
