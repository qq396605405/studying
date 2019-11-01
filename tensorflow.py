#训练一个全连接神经网络
import tensorflow as tf
import math
import numpy as np
import types
import matplotlib.pyplot as plt
import pylab
#定义一个y=sin(x)的曲线
def draw_correct_line():
    x=np.arange(0,2*np.pi,0.01)
    x=x.reshape(len(x),1)
    y=np.sin(x)
    pylab.plot(x,y,label='标准sin曲线')
    plt.axhline(linewidth=1,color='r')

#返回一个训练样本
def get_train_data():
    train_x=np.random.uniform(0.0,2*np.pi,(1))
    train_y=np.sin(train_x)
    return train_x,train_y


#定义网络结构
def inference(input_data):
    with tf.variable_scope('hidden1'):
    #第一个隐藏层，采用16个隐藏点
        weights=tf.get_variable('weight',[1,16],tf.float32,initializer=tf.random_normal_initializer(0.0,1))
        biases=tf.get_variable('bias',[1,16],tf.float32,initializer=tf.random_normal_initializer(0.0,1))
        hidden1=tf.sigmoid(tf.multiply(input_data,weights)+biases)

    with tf.variable_scope('hidden2'):
    #第二个隐藏层，采用16个隐藏点
        weights=tf.get_variable('weight',[16,16],tf.float32,initializer=tf.random_normal_initializer(0.0,1))
        biases=tf.get_variable('bias',[16],tf.float32,initializer=tf.random_normal_initializer(0.0,1))
        hidden2=tf.sigmoid(tf.matmul(hidden1,weights)+biases)

    with tf.variable_scope('hidden3'):
    #第三个隐藏层，采用16个隐藏点
        weights=tf.get_variable('weight',[16,16],tf.float32,initializer=tf.random_normal_initializer(0.0,1))
        biases=tf.get_variable('bias',[16],tf.float32,initializer=tf.random_normal_initializer(0.0,1))
        hidden3=tf.sigmoid(tf.matmul(hidden2,weights)+biases)

    with tf.variable_scope('output_layer'):
    #第二个隐藏层，采用16个隐藏点
        weights=tf.get_variable('weight',[16,1],tf.float32,initializer=tf.random_normal_initializer(0.0,1))
        biases=tf.get_variable('bias',[1],tf.float32,initializer=tf.random_normal_initializer(0.0,1))
        output=tf.matmul(hidden3,weights)+biases
    return output

def train():
    learning_rate=0.01
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    net_out=inference(x)
    #定义损失函数
    loss_op=tf.square(net_out-y)
    #采用随机梯度下降的优化函数
    opt=tf.train.GradientDescentOptimizer(learning_rate)
    train_op=opt.minimize(loss_op)
    init=tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print('start training')
        for i in range(1000000):
            train_x,train_y=get_train_data()
            sess.run(train_op, feed_dict={x:train_x,y:train_y})

            if i%10000==0:
                times=int(i/10000)
                test_x_ndarray=np.arange(0,2*np.pi,0.01)
                test_y_ndarray=np.zeros([len(test_x_ndarray)])
                ind=0
                for test_x in test_x_ndarray:
                    test_y=sess.run(net_out,feed_dict={x:test_x,y:1})
                    np.put(test_y_ndarray,ind,test_y)
                    ind+=1

                draw_correct_line()
                pylab.plot(test_x_ndarray,test_y_ndarray,'--',label=str(times)+'times')
if __name__=="__main__":
    train()



#实现CNN卷积神经网络
import tensorflow as tf
import os
import cv2
tf_writer=tf.python_io.TFRecordWriter(path='train.tfrecords')
for file in os.listdir(path='data_dir'):
    file_path=os.path.join('data_dir',file)
    image_data=cv2.imread(file_path)
    image_bytes=image_data.tostring()
    #图片的高
    rows=image_data.shape[0]  #第一维度的长度    这些可以通过image_data.shape来查看
    #图片的宽
    cols=image_data.shape[1]
    #图片通道数
    channels=image_data.shape[2]
    label_data=0
    example=tf.train.Example()
    feature=example.features.feature   #相当于数据放入feature中，一个映射
    feature['height'].int64_list.value.append(rows)
    feature['width'].int64_list.value.append(cols)
    feature['channels'].int64_list.value.append(channels)
    feature['image_data'].bytes_list.value.append(image_bytes)
    feature['label'].int64_list.value.append(label_data)
    tf_writer.write(example.SerializeToString())
tf_writer.close()

#构造卷积神经网络结构
def weight_init(shape,name):   #shape参数的形状
    return tf.get_variable(name,shape,initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1))

def bias_init(shape,name):
    return  tf.get_variable(name,shape,initializer=tf.constant_initializer(0.0))

def conv2d(x,conv_w):
    return tf.nn.conv2d(x,conv_w,strides=[1,1,1,1],padding='VALID')

def max_pool(x,size):
    return tf.nn.max_pool(x,ksize=[1,size,size,1],strides=[1,size,size,1],padding='VALID')

def inference(input_data):
    with tf.name_scope('conv1'):   #指定的区域中定义的所有对象及各种操作，他们的“name”属性上会增加该命名区的区域名，用以区别对象属于哪个区域；
        w_conv1=weight_init([10,10,3,16],'conv1_w')   #卷积核大小是10X10，输入是3通道，输出是16通道
        b_conv1=bias_init([16],'conv1_b')
        h_conv1=tf.nn.relu(conv2d(input_data,w_conv1)+b_conv1)  #卷积过程
        #池化
        h_pool1=max_pool(h_conv1,2)
    with tf.name_scope('conv2'):
        #卷积核大小是5X5，输入是16通道，输出是16通道
        w_conv2=weight_init([5,5,16,16],'conv2_w')
        b_conv2=bias_init([16],'conv2_b')
        #卷积之后，图片大小变成41X41（45-5+1=41）
        h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
        h_pool2=max_pool(h_conv2,2)
    with tf.name_scope('conv3'):
        #卷积核大小是5X5,输入是16通道，输出16通道
        w_conv3=weight_init([5,5,16,16],'conv3_w')
        b_conv3=bias_init([16],'conv3_b')
        h_conv3=tf.nn.relu(conv2d(h_pool2,w_conv3)+b_conv3)
        h_pool3=max_pool(h_conv3,2)

    with tf.name_scope('fc1'):
        w_fc1=weight_init([8*8*16,128],'fc1_w')
        b_fc1=bias_init([128],'fc1_b')
        h_fc1=tf.nn.relu(tf.matmul(tf.reshape(h_pool3,[-1,8*8*16]),w_fc1)+b_fc1)

    with tf.name_scope('fc2'):
        w_fc2=weight_init([128,2],'fc2_w')
        b_fc2=bias_init([2],'fc2_b')
        h_fc2=tf.matmul(h_fc1,w_fc2)+b_fc2

    return h_fc2

def read_and_decode(filename_queue):
    #读取TFRecord文件队列数据，解码成张量
    reader=tf.TFRecordReader()
    #从文件队列中读取数据
    _,serialized_example=reader.read(filename_queue)
    #数据反序列化为结构化的数据
    features=tf.parse_single_example(
        serialized_example,
        features={
            'height':tf.FixedLenFeature([],tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'channels': tf.FixedLenFeature([], tf.int64),
            'image_data': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    #将image_data部分的数据解码成张量
    image=tf.decode_raw(features['image_data'],tf.unit8)
    #将image_data的tensor变成100X100的大小，3通道
    image=tf.cast(image,tf.float32)
    label=tf.cast(features['label'],tf.int32)
    return image,label

def inputs(filename,batch_size):
    #读取TFRecord文件数据tensorflow中可以计算的张量数据
    with tf.name_scope('input'):
        #生成文件队列，最多迭代2000次
        filename_queue=tf.train.string_input_producer([filename],num_epochs=2000)
        #从文件中读入数据，并且变成张量格式
        image,label=read_and_decode(filename_queue)
        #将数据按组大小返回，并且随机打乱

def train():
    #定义一个global_step的张量，在训练过程中记录训练的步数
    global_step=tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False,dtype=tf.int32)
    from absl import flags
    from absl.flags import FLAGS
    batch_size=flags.FLAGS.batch_size
    #读取之前生成的TFRecord文件，得到训练和验证的数据
    train_images,train_labels=input("./tfrecord_data/train,tfrecord",batch_size)
    test_images, test_labels = input("./tfrecord_data/train,tfrecord", batch_size)
    #将label的数据变成one_hot的形式，作用是将一个值化为一个概率分布的向量，一般用于分类问题。
    train_labels_one_hot=tf.one_hot(train_labels,2,on_value=1.0,off_value=0.0)
    test_labels_one_hot = tf.one_hot(train_labels, 2, on_value=1.0, off_value=0.0)
    learning_rate=0.0000001
    with tf.variable_scope('inference') as scope:
        #进行向前计算
        train_y_conv=inference(train_images)
        #这个变量空间下的变量是按照变量名字共享的
        scope.reuse_variables()
        test_y_conv=inference(test_images)
        #计算sofrmax损失值
        cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_labels_one_hot,
                                                                             logits=train_y_conv))
        #优化函数采用普通的随机梯度下降优化算法
        optimizer=tf.train.GradientDescentOptimizer(learning_rate)
        #每执行一次训练，参数global_step会增加1
        train_op=optimizer.minimize(cross_entropy,global_step=global_step)

        #计算训练集的准确率
        train_correct_prediction=tf.equal(tf.argmax(train_y_conv,1),    #tf.argmax返回每一行最大元素的位置
                                          tf.argmax(train_labels_one_hot,1))
        train_accuracy=tf.reduce_mean(tf.cast(train_correct_prediction,tf.float32))
        #计算验证集的准确率
        test_correct_prediction=tf.equal(tf.argmax(test_y_conv,1),
                                         tf.argmax(test_labels_one_hot,1))
        test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))

        #初始化参数
        init_op=tf.global_variables_initializer()
        local_init_op = tf.global_variables_initializer()

        #定义保存和装载模型参数的保存器
        saver=tf.train.Saver()

        #记录训练过程中的损失值和训练数据的准确率
        tf.summary.scalar('cross_entrop_loss',cross_entropy)
        tf.summary.scalar('train_acc', train_accuracy)
        summary_op=tf.summary.merge_all()

        #设置GPU利用率
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=flags.FLAGS.gpu_memory_fraction)
        config=tf.ConfigProto(gpu_options=gpu_options)

        with tf.Session(config=config) as sess:
            #是否重新训练还是从保存的模型中装载参数
            if FLAGS.reload_model==1:
                ckpt=tf.train.get_checkpoint_state(FLAGS.model_dir)
                saver.restore(sess,ckpt.model_checkpoint_path)
                save_step=int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                print("reload model from %s,save_step = %d" %(ckpt.model_checkpoint_path,save_step))
            else:
                print('Create model with fresh paramters')
                sess.run(init_op)
                sess.run(local_init_op)

            #定义记录summary的writer
            summary_writer=tf.summary.FileWriter(FLAGS.event_dir,sess.graph)
            #负责处理读取数据过程中的异常，比如负责清理关闭的线程
            coord=tf.train.Coordinator()
            #调用start_queue_runners开始读数据，防止训练堵塞
            threads=tf.train.start_queue_runners(sess=sess,coord=coord)
            try:
                while not coord.should_stop():
                    #执行训练过程，会执行前向计算和反向传播的参数更新
                    _,g_step=sess.run([train_op,global_step])
                    if g_step%2==0:
                        #记录训练过程损失值和训练数据的准确率
                        summary_str=sess.run(summary_op)
                        summary_writer.add.summary(summary_str,g_step)
                    if g_step%100==0:
                        train_accuracy_value,loss=sess.run([train_accuracy,cross_entropy])
                        print("step %d training_acc is %.2f,loss is %.4f"%(g_step,train_accuracy_value,loss))
                    if g_step%1000==0:
                        test_accuracy_value=sess.run(test_accuracy)
                        print("step %d test_acc is %.2f"%(g_step,test_accuracy_value))
                    if g_step%2000==0:
                        print("save model to %s"% FLAGS.model_dir+"model.ckpt."+str(g_step))
                        saver.save(sess,FLAGS.model_dir+"model.ckpt",global_step=global_step)
            except tf.errors.OutOfRangeError:
                pass
            finally:
                coord.request_stop()
            coord.join(threads)





