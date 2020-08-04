using GR
using Printf

# 参数
epsilon = 0.1
state_cake=[3,3]
state_poison=[[2,3],[3,2]]
discount_factor=0.9
learning_rate=0.01

# Q-table
qtable=Dict([[i,j]=>[0.0,0.0,0.0,0.0] for i=1:5,j=1:5])

# 读入图片
wm, hm, mouse = readimage("mouse.png")
wp, hp, poison = readimage("poison.png")
wc, hc, cake = readimage("cake.png")

# state to coordinate
function s2c(state::Array{Int64,1})
    xmin=(state[1]-1)*0.2
    xmax=state[1]*0.2
    ymin=(state[2]-1)*0.2
    ymax=state[2]*0.2
    return xmin,xmax,ymin,ymax
end

#绘制游戏网格，m为格子的行数，n为格子的列数。
function render(state::Array{Int64,1})
    setwindow(-0.01,1.01,-0.01,1.01)
    setviewport(0,1,0,1)
   
    clearws()
    setwswindow(0,1,0,1)
    # 水平线
    for y in range(0,stop=1,length=6)
        polyline([0,1],[y,y])
    end
    # 垂直线
    for x in range(0,stop=1,length=6)
        polyline([x,x],[0,1])
    end
    # 画毒药    
    xmin,xmax,ymin,ymax=s2c(state_poison[1])
    xmin+=0.01
    xmax-=0.01
    ymin+=0.01
    ymax-=0.01
    drawimage(xmin,xmax,ymin,ymax,wp,hp,poison)
    xmin,xmax,ymin,ymax=s2c(state_poison[2])
    xmin+=0.01
    xmax-=0.01
    ymin+=0.01
    ymax-=0.01
    drawimage(xmin,xmax,ymin,ymax,wp,hp,poison)
    #画蛋糕
    xmin,xmax,ymin,ymax=s2c(state_cake)
    xmin+=0.01
    xmax-=0.01
    ymin+=0.01
    ymax-=0.01
    drawimage(xmin,xmax,ymin,ymax,wc,hc,cake)
    # 画老鼠
    xmin,xmax,ymin,ymax=s2c(state)
    xmin+=0.01
    xmax-=0.01
    ymin+=0.01
    ymax-=0.01
    drawimage(xmin,xmax,ymin,ymax,wm,hm,mouse)
    # 绘制q表
    setcharheight(0.018)
    setcharup(0,1)
    for q in qtable
        xmin,xmax,ymin,ymax=s2c(q[1])
        text(xmin+0.06,ymax-0.03,@sprintf("%.2f",qtable[q[1]][1]))
        text(xmin+0.06,ymin+0.01,@sprintf("%.2f",qtable[q[1]][2]))       
    end
    setcharup(-1,0)
    for q in qtable
        xmin,xmax,ymin,ymax=s2c(q[1])
        text(xmin+0.03,ymin+0.06,@sprintf("%.2f",qtable[q[1]][3]))
        text(xmax-0.01,ymin+0.06,@sprintf("%.2f",qtable[q[1]][4]))       
    end
    
    updatews()
end

function get_action(state::Array{Int64,1})
    actions=[1,2,3,4]
    state[2]>4 && deleteat!(actions,findfirst(isequal(1),actions))
    state[2]<2 && deleteat!(actions,findfirst(isequal(2),actions))
    state[1]<2 && deleteat!(actions,findfirst(isequal(3),actions))
    state[1]>4 && deleteat!(actions,findfirst(isequal(4),actions))

    v=qtable[state]
    if rand()<epsilon || v[1]==v[2]==v[3]==v[4] 
        action=rand(actions)
    else
        action=argmax(v)
    end
    return action
end

function step(state::Array{Int64,1},action::Int)
    next_state=copy(state)
    if action==1 #up
        state[2]<5 && (next_state[2]+=1)
    elseif action==2 #down
        state[2]>1 && (next_state[2]-=1)
    elseif action==3 #left
        state[1]>1 && (next_state[1]-=1)
    else
        state[1]<5 && (next_state[1]+=1)
    end
    
    if next_state==state_cake
        reward=100
        done=true
    elseif next_state in state_poison
        reward=-100
        done=true
    else
        reward=0
        done=false
    end

    return next_state,reward,done
end

function learn(state::Array{Int64,1},action::Int,reward::Real,next_state::Array{Int64,1})
    current_q = qtable[state][action]
    # 贝尔曼方程更新
    new_q = reward + discount_factor * maximum(qtable[next_state])
    qtable[state][action] += learning_rate * (new_q - current_q)
end

function train(n_episode::Int)
    for episode in 1:n_episode        
        state = [1,1]
        done=false
        while done==false 
            render(state)                
            action = get_action(state)
            next_state, reward, done = step(state,action)
            # 更新Q表
            learn(state,action,reward,next_state)
            state = next_state            
        end
    end
end

function play()
    state = [1,1]
    done=false
    reward=0
    while done==false
        sleep(0.5) 
        render(state)                
        action = argmax(qtable[state])
        next_state,reward,done = step(state,action)
        state = next_state            
    end
    println(reward)
end

@info "开始学习..."
train(1000)
@info "学习结束，开始演示"
play()