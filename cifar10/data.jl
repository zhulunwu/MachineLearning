using Flux

const data_root=Sys.iswindows() ? "D:/" : "/mnt"
const colorfactor=Float32(1.0/255.0)

# 训练集,5个文件，i不超过5.
function readDataset(i::Int)
    file=data_root*"/Explore/DataSets/cifar-10-batches-bin/data_batch_$(i).bin"
    data_bytes=open(file,"r") do io
        [(read(io,UInt8),read(io,32*32*3)) for i=1:10000]
    end
    return data_bytes
end

function train_image_label(i::Int)
    label_image_bytes=readDataset(i)
    image_bytes=map(last,label_image_bytes);
    channelwise=map(x->reshape(x,(1024,3)),image_bytes); 
    images=map(channelwise) do x
        r=reshape(transpose(reshape(x[:,1],(32,32))),(32,32,1)) 
        g=reshape(transpose(reshape(x[:,2],(32,32))),(32,32,1))
        b=reshape(transpose(reshape(x[:,3],(32,32))),(32,32,1))
        cat(r,g,b,dims=3)
    end
    train_images=map(x->Float32.(x)*colorfactor,images) #最终的训练图像数据    
    train_labels=map(first,label_image_bytes)
    return train_images,train_labels
end

function test_image_label()
    # 测试集
    file=data_root*"/Explore/DataSets/cifar-10-batches-bin/test_batch.bin"
    test_bytes=open(file,"r") do io
        [(read(io,UInt8),read(io,32*32*3)) for i=1:10000]
    end

    image_test_bytes=map(last,test_bytes);
    channelwise_test=map(x->reshape(x,(1024,3)),image_test_bytes);
    imgs_test=map(channelwise_test) do x
        r=reshape(transpose(reshape(x[:,1],(32,32))),(32,32,1)) 
        g=reshape(transpose(reshape(x[:,2],(32,32))),(32,32,1))
        b=reshape(transpose(reshape(x[:,3],(32,32))),(32,32,1))
        cat(r,g,b,dims=3)
    end
    test_images=map(x->Float32.(x)*colorfactor,imgs_test)
    test_labels=map(first,test_bytes) # 测试标签
    return test_images,test_labels
end
