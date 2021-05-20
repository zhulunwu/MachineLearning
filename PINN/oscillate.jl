# 简谐振动的学习
#= 网络要求：输入时间，输出位移。不过考虑到zygote求导阶数的限制，需要将二阶微分方程分解为两个一阶微分方程。
要对t求导，t显然是必须的。
简谐振动的微分方程组为：dx/dt=v以及dv/dt=-ω^2*x。这暗示我们，应该输出x和v。由于两个输出至少有两个神经元处理，所以不会产生混淆。
以此类推，高阶微分方程可以降为多个一阶微分方程来处理。
=#
using Flux
import Flux:mse
import Flux:train!

# 构建网络
m=Chain(Dense(1,10,tanh),Dense(10,10,tanh),Dense(10,10,tanh),Dense(10,2))
x(m,t)=sum(m(t).*[1,0]) 
v(m,t)=sum(m(t).*[0,1])
dxdt(m,t)=last(gradient(x,m,t))[1]
dvdt(m,t)=last(gradient(v,m,t))[1]

# 简谐振子的参数和初始条件,k=1,m=1,ω=1.
x0=0.5
v0=0

# 损失函数
loss(t)=mse([dxdt(m,t),dvdt(m,t)],[v(m,t),-x(m,t)])+mse(v(m,t)^2+x(m,t)^2,x0^2)+mse(m([0]),[x0,v0])

# 训练
for i=1:4000
    t=rand(1,100)*10
    train!(loss,params(m),[t],ADAM())
    i%100==0 && print("\rloss=",loss((0:0.1:10)'))
end

# 测试
using Makie
t=0:0.1:10
y=m(t')[1,:]
z=m(t')[2,:]
E=y.^2+z.^2
vbox(lines(t,y),lines(t,z),lines(t,E))
# 貌似很难训练，效果很差。
