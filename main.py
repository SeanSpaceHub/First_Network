import numpy as np
import pickle

filename = 'w&b.txt'

from Reader_for_minist import *
from matplotlib import pyplot as plt
from DNN import *

#下一步: 使用drop（Over）/深度学习/调参数/分层实验/分层调查/momemtum方法/cnn/
if __name__ == '__main__':

    Learn_speed = 0.001
    batch_size = 250
    learn_roop = 1000
    dropout_rate = 0.75  #神经元的数目需要可以被乘尽
    wide = 2
    learn_times = 6000 // batch_size * 8
    memory_size = 1500

    #decay = 0.00001  #0.00001好像还可以到0.8,但精度还是容易掉下去0.8 -> 0.7,虽然会回来,收敛在0.7就很慢了 #先不要使用,让训练不稳定???  #不可太大,会让收敛困难 0.0001 -> 0.97 , 0.001 -> 0.96 ,0.01 ->0.3
    #Netstruct = (batch_size, 28 * 28, 28, 28, 28*2, 28*3, 28*2, 20, 10) #可以到0.75
    #Netstruct = (batch_size, 28 * 28, 28 * 14, 28 * 7, 28 * 3, 28, 10) #可以到0.82 #很慢 #也是最后慢慢收敛
    #Netstruct = (batch_size, 28 * 28, 200, 150, 100, 50, 10) #0.8 #5层
    #Netstruct = (batch_size, 28 * 28, 100, 200, 150, 150, 100, 100, 100, 50, 50, 50, 40, 40, 20, 10, 10) #效果没有提升（甚至降低），且收敛太慢  #应该是参数的问题
    #Netstruct = (batch_size, 28 * 28, 40, 100, 200, 400, 100, 100, 200, 40, 40, 10) #10层
    Netstruct = (batch_size, 28 * 28, 100, 160, 100, 100, 40, 40, 10)
    ##ReLU or Sigmoid
    trainer = Neural_Network(Affine, ReLU, Softmax_with_loss, Netstruct, wide, dropout=True)

    train_images = load_train_images() / 256.0
    train_labels = turn_label(load_train_labels().reshape((-1, 1)))
    test_images = load_test_images() / 256.0
    test_labels = turn_label(load_test_labels().reshape((-1, 1)))

    generator_x = batcher_random(train_images, train_labels, 784, 10, batch_size)

    generator_x_test = batcher(test_images, 784, batch_size)
    generator_t_test = batcher(test_labels, 10, batch_size)

    accuracy_trainer = np.array([0, ])
    accuracy_batch = []
    accuracy_batch_t = []
    loss_batch = []
    i = 0
    x = None
    t = None
    x_t = next(generator_x_test)
    t_t = next(generator_t_test)
    testor = Neural_Network(Affine, ReLU, Softmax_with_loss, Netstruct)

    while i < learn_times:
        #learn_roop= int(learn_roop*0.8)    #递减的训练次数
        x, t = next(generator_x)
        n = 0
        while n < learn_roop:
            testor.reload_parameter(trainer.get_parameter())
            testor.forward(x_t, t_t)
            accuracy_batch_t.append(testor.get_accuracy())

            #trainer.forward_with_weightdecay(x, t, decay)
            #trainer.backward_with_weightdecay()
            #save_Y = trainer.forward(x, t, save_Y=True)
            #plt_sub((plt_distribution,), save_Y)#, x_limit=(-1, 1), y_limit=(0, 0.5))
            #save_dx = trainer.backward(save_dx=True)
            #plt_sub((plt_distribution,), save_dx, x_limit=(-100, 100), y_limit=(0, 0.5))

            trainer.refresh_dropout(dropout_rate)

            #print(np.sum(trainer.parameter['X1']))
            #print(trainer.parameter['W0'].shape)
            #print(trainer.layer['Affine0'].W.shape)
            trainer.forward(x, t)
            trainer.backward()
            #trainer.updata(Learn_speed)  #用这个就是学不进
            trainer.updata_AdaGrad(Learn_speed, memory_size)
            accuracy_trainer.append(trainer.get_accuracy())
            #trainer.updata_numerical(Learn_speed)
            n += 1
        #plt_change_process((accuracy_trainer, trainer.get_loss()), ('accuracy', 'loss'))
        print(accuracy_trainer)
        print(np.sum(accuracy_trainer)/batch_size)
        accuracy_batch.append(np.sum(accuracy_trainer)/batch_size)
        accuracy_trainer = []
        loss_batch.append(trainer.get_loss()[0])


        #print(trainer.get_loss())
        i += 1

    plt_change_process((accuracy_batch, accuracy_batch_t), ('trainer', 'testor'))

    save_Y = trainer.forward(x, t, save_Y=True)
    plt_sub((plt_distribution,), save_Y)#, x_limit=(-1, 1), y_limit=(0, 0.5))






