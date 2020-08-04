# 对卷积算法进行验证
using Flux
# 准备一个4*4的矩阵，后面两个1是数据格式的需要。WHCN格式，即宽度，高度、通道数，批量
# 此处考虑灰度图，只有一个通道。
data=rand(4,4,1,1)
# 一个简单的卷积层，
m=Conv((3,3),1=>1) 
ps=params(m)
w=dropdims(ps.order[1],dims=(3,4))
b=ps.order[2] #默认偏执为0
rw=reverse(reverse(w,dims=1),dims=2) 
o11=sum(data[1:3,1:3].*rw)
o12=sum(data[1:3,2:4].*rw)
o21=sum(data[2:4,1:3].*rw)
o22=sum(data[2:4,2:4].*rw)
out=[o11 o12;o21 o22]
m(data)

# 通道数量计算
m=Conv((3,3),1=>2)
m(data)
ps=params(m)
ps.order[1]
# 可以发现多了一个通道后，weights 增加了一套，这就好理解了。

# 模拟RGB三通道数据
data=rand(3,3,3,1)
m=Conv((3,3),3=>2)
ps=params(m)
w=ps.order[1] #共有6套权值，输入RGB三个画面，每个画面有一个3*3。每个分量图片有两个输出，所以再翻一倍。
wr=[reverse(reverse(w[:,:,i,j],dims=1),dims=2) for i=1:3,j=1:2]

sum(sum([data[:,:,i,1].*wr[i,1] for i=1:3])) #相当于再经过一个神经元完成求和过程
sum(sum([data[:,:,i,1].*wr[i,2] for i=1:3])) #最终只有两个输出
m(data)

# 池化算法验证
data=rand(4,4,1,1)
m=MaxPool((2,2))
m(data)
# 直接观察data和池化输出的结果即可看出，不过是从中取最大值而已。

# 关于pad参数
m=Conv((3,3),1=>1)
mp=Conv((3,3),1=>1,pad=(2,2))
m(data)
mp(data)
# 看起来pad=(2,2)即上下左右各补2行（列）零，这样导致卷积后的矩阵大小发生变化。
# 也支持不对成补0，比如pad=(0,1,1,1),这样左边就不补0。