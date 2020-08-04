# 生成对抗网络的简单范例
- v0版本
gpu训练，使用了卷积层。
计算量比较大，计算结果因计算量大而暂时不知。

- v1版本
多层感知机模型，模型的参照代码为：
https://github.com/wiseodd/generative-models/tree/master/GAN/wasserstein_gan
没有使用Flux自带的train函数。

- v2版本
模型同v1版本，但是使用了Flux自带的train函数。

- v3版本
同v0版本，但是参数略小，笔记本显卡可以计算。

- 一些记录
最先根据python的模型代码编写训练无果，后加上零梯度代码后可以训练。需要从算法上理解这一现象，并了解
Flux和tensorflow的差异。

