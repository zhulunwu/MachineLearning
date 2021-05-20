using Flux

w=rand() #神经网络的权值
x=rand() #输入数据
u(x,w)=sigmoid(w*x) #神经网络输出
du(x,w)=gradient(u,x,w)[1] #du/dx
loss(x,w)=u(x,w)+du(x,w) # 损失函数
dw=gradient(loss,x,w)[2]
w-=dw*0.01 #调整参数