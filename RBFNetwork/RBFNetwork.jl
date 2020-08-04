using LinearAlgebra
using GR
# 一维数据
x=range(0,stop=10,length=100)
y=sin.(x)

σ=1.0
hidden=10
centers=rand(x,hidden)

kernel_function(c,p)=exp(-σ*(norm(c-p))^2)

G=[kernel_function.(centers[i],x) for i=1:hidden]
G=hcat(G...)

iG=pinv(G)
weights=[dot(iG[i,:],y) for i=1:hidden]

Y=[dot(G[i,:],weights) for i=1:length(x)]
plot(x,y,x,Y)