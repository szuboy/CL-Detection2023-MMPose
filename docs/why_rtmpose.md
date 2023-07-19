<img alt="avatar" src="./images/banner.jpeg"/>

# 为什么我们选择 `RTMPose` 框架

**敲黑板**：RTMPose 框架的基本资料：[arXiv:2303.07399](https://arxiv.org/abs/2303.07399)
| [项目页面](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)
| [官方知乎解读](https://zhuanlan.zhihu.com/p/613258581)

进来看到这里的小伙伴，丝毫不需要慌张，如果您计划使用 `RTMPose` 框架，还请记得安装 MMDetection（用于骨干网络配置） 和 Albumentations （训练数据扩增）包，可执行下面的安装命令（如果仅使用 MMPose，可忽略这两个依赖的安装）：
```
pip install albumentations
mim install "mmdet==3.0.0"
```

笔者会在这里尽可能回答您可能存在的疑问，我将会结合 [RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose
](https://arxiv.org/abs/2303.07399) 和一些大佬的解读对配置文件中的操作进行详细解读：




----

1. RTMPose 框架有哪些优势？
- 通过一些有针对性的选择和设计，RTMPose 的优势在于：速度快 + 定位准，对落地部署非常地友好；无论是学术科研刷比赛，还是产业部署做横向，都可以去参考这个框架内部的一些做法。如果您想继续了解，还请继续往下看。

----

2. RTMPose 框架采用的是哪一种关键点定位的范式？

- RTMPose 框架采用的是自上而下 `TopdownPoseEstimator` 的实验范式，论文中认为当前的检测网络的性能不再是性能的瓶颈，在大部分场景下都能够实时地进行多个对象的关键点定位，具体的体现在配置文件这里：
  ```python
  # model settings
  model = dict(
    type='TopdownPoseEstimator',
    ...
  )
  ```

----

3. RTMPose 框架基于哪一个骨干网络？
- RTMPose 默认采用的是 `MMDetection` 框架的 `CSPNeXt` 骨干网络，以平衡速度和精度，并兼顾部署的友好性。虽然...，但是...，我不太理解为嘛是 `MMDetection` 下的的骨干网络，为什么要和其他框架耦合在一起，不负责任的猜测：因为是 `OpenMMLab` 全家桶的成员，所以拿过来就用了，没有考虑那么多。如果您要自定义配置骨干网络，修改下面这段代码即可：
  ```python
    model = dict(
    ...
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth')),
    ...
    ) 
  ```

----

4. RTMPose 框架的对于特征的解码 Head 的设计有什么特别，这会带来什么好处？
- RTMPose 引入了一个自上而下的头部：由大核卷积层、全连接层和门控注意力单元组成，用于从低分辨率特征图生成一维表示。
  是的，这和大家常用的二维热图作为监督信号不同，RTMPose 为了轻量化，采用了 SimCC 的方式对关键点进行监督，将关键点看成是两个一维的信号进行监督。如果您不了解 SimCC，可参考镜佬的这篇[知乎解读](https://zhuanlan.zhihu.com/p/451958229)，相信您看过会有一股恍然大明白的感觉。
  ```python
  head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=38,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec)
  ```
- 值得注意到：由于 CL-Detection2023 挑战赛的图像是还是比较大的（大概约是大小是宽高约为2000个像素），正如镜佬在[知乎解读](https://zhuanlan.zhihu.com/p/451958229)中的那样，大尺度的情况下，或许热图回归带来的性能收益会更大，这也是为什么 RTMPose 基线模型性能稍稍低于 HRNet 热图回归的基线模型。
当然啦，目前的比较是不公平的，仅只能仅作参考，大家也可以继续对 RTMPose 进行优化和调参，看看性能上限到哪里，或许这就是解决方案的胜出的点。
----

5. 上面的关于模型的部分已经定义完成，其他的训练和测试配置可以 MMPose 框架相通。当然啦，如果您还有什么疑问，请大方地在 Issue 提出来，我会第一时间出来解答滴~
















