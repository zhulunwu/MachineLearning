# 使用svd方法进行图像简单降噪处理
using Images,Colors
import LinearAlgebra:svd,Diagonal

# 首先读入图像
img=load("girl.bmp")
# 转变为灰度图
img=Gray.(img)
# 将图像的数据变为常规可以处理的数据格式
imgdata=float(channelview(img))
# svd分解
u,s,v=svd(imgdata)
# 需要保留的奇异值的数量
k=50
# 根据所选的奇异值重构图像矩阵
M = u[:, 1:k] * Diagonal(s[1:k]) * v[:, 1:k]'
# 图像值标准化
M = min.(max.(M, 0.0), 1.0)
# 数据转图像
colorview(Gray,M)
# 后记：其实效果并不好，只是演示一下原理。