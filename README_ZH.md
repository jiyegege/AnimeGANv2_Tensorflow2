# AnimeGANv2_Tensorflow2
参考源 [AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGAN) 项目，用TensorFlow2去改写实现

## 安装和测试环境

GPU：3060 batch_size=10 训练耗时为11min/epoch

- tensorflow==2.8.0
- tensorflow-addons==0.16.1
- wandb
- tqdm==4.63.1
- PyYAML
- opencv-python==4.5.5

## 使用

### 训练

```shell
python train.py --config_path config/config-defaults.yaml --dataset Hayao --hyperparameters False
```

- `--config_path`配置文件路径默认在`config/config-defaults.yaml`下面，里面是项目的超参数配置
- `--dataset`数据集的名字
- `--hyperparameters`是否启用wandb的超参数搜索
- `--pre_train_weight`预训练权重，可加载以前训练好的模型权重进行微调后再训练成新的模型

### 测试

## 训练过程

### loss变化

判别器相关loss

![image-20220624202624359](doc_pic/D_all_loss.png)



生成器相关loss

![image-20220624202832128](doc_pic/G_all_loss.png)



生成器和判别器的loss相对变化

![image-20220624202944846](doc_pic/D_G_loss.png)

由loss可见，生成器和判别器产生了明显的对抗效果，生成器的loss成上升趋势，判别器loss成下降趋势，由于训练的相关loss权重是按照原作者推荐的方式来的，与原作者训练的效果有一定出入，需要自己再进行调整

### 图片验证结果

![图片1](doc_pic/pic.png)

![pic2](doc_pic/pic2.png)

![pic3](doc_pic/pic3.png)

图片效果整体动漫风格相对于原作者较强，但是图片细节缺失较多，图片色彩较强，需要通过调整loss权重进一步改善


## License
- 此版本仅用于学术研究和非商业用途，如果用于商业目的，请联系我以获得授权批准