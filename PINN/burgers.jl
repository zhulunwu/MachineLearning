using Flux
import Flux:mse,Zygote
using MAT

# 读入数据
data=matread("burgers_shock.mat")
x=Float32.(data["x"])
t=Float32.(data["t"])
usol=Float32.(data["usol"])

# 构造训练数据[x,t]和u
ix=rand(1:length(x),2000)
it=rand(1:length(t),2000)
xt_train=[[x[ix[n]],t[it[n]]] for n=1:length(it)]
u_train=[usol[ix[n],it[n]] for n=1:length(it)]
data_train=zip(xt_train,u_train)

# 构造神经网络[2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
λ1=rand()
λ2=rand()
m=Chain(Dense(2,2,tanh),Dense(2,1))
u(m,xt)=sum(m(xt))
ux(m,xt)=gradient(u,m,xt)[2][1]
ut(m,xt)=gradient(u,m,xt)[2][2]
uxx(m,xt)=gradient(ux,m,xt)[2][1]
f(m,xt)=ut(m,xt)+λ1*u(m,xt)*ux(m,xt)-λ2*uxx(m,xt)
loss(m,xt,us)=mse(u(m,xt),us)+mse(f(m,xt),0)
gradient(()->loss(m,xt,us),params(m))

#训练神经网络
for (xt,us) in data
    ps=Flux.Params(params(m))
    gs = gradient(ps) do
        loss(m,xt,us)
    end
    Flux.update!(ADAM(),ps, gs)
end