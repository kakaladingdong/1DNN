import tensorflow as tf
import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt

# import element_p.element.element as fusion
import attention.data_loader
import attention.active_function as af
class Model():

    def __init__(self,args,sess):
        self.sess = sess
        self.result = args.result
        info = sio.loadmat(os.path.join(self.result,'info.mat'))
        self.shape = info['shape']
        self.dim = info['dim']
        self.class_num = int(info['class_num'])
        self.data_gt = info['data_gt']
        self.log = args.log
        self.model = args.model
        self.data_path = args.data_path
        self.epoch = args.epoch

        self.tfrecords=args.tfrecords
        self.global_step = tf.Variable(0,trainable=False)

        if args.use_lr_decay:
            self.lr = tf.train.exponential_decay(learning_rate=args.lr,
                                             global_step=self.global_step,
                                             decay_rate=args.decay_rete,
                                             decay_steps=args.decay_steps)
        else:
            self.lr = args.lr

        self.image = tf.placeholder(dtype=tf.float32, shape=(None, self.dim))
        self.label = tf.placeholder(dtype=tf.int64, shape=(None, 1))

        # self.classifer = self.classifer

        self.pre_label = self.classifer(self.image)
        self.model_name = os.path.join('model.ckpt')
        self.loss()
        self.summary_write = tf.summary.FileWriter(os.path.join(self.log),graph=self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=100)

    def loss(self):
        with tf.variable_scope('loss'):
            loss_cross_entropy = tf.losses.sparse_softmax_cross_entropy(self.label,self.pre_label,scope='loss_cross_entropy')
            loss_cross_entropy = tf.reduce_mean(loss_cross_entropy)
            self.loss_total = loss_cross_entropy
            tf.summary.scalar('loss_total',self.loss_total)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss_total,global_step=self.global_step)
        self.merged = tf.summary.merge_all()


    def classifer(self,feature):
        feature = tf.expand_dims(feature,2)
        f_num = 16
        print(feature)
        with tf.variable_scope('classifer',reuse=tf.AUTO_REUSE):
            with tf.variable_scope('conv0'):
                conv0 = tf.layers.conv1d(feature,f_num,(8),strides=(3),padding='valid')
                conv0 = tf.layers.batch_normalization(conv0)
                conv0=tf.nn.relu(conv0)
                conv0=af.x_unit(conv0)
                print(conv0)
            with tf.variable_scope('conv1'):
                conv1 = tf.layers.conv1d(conv0,f_num*2,(3),strides=(2),padding='valid')
                conv1 = tf.layers.batch_normalization(conv1)
                conv1=tf.nn.relu(conv1)
                conv1=af.x_unit(conv1)
                print(conv1)
            with tf.variable_scope('conv2'):
                conv2 = tf.layers.conv1d(conv1,f_num*4,(3),strides=(2),padding='valid')
                conv2 = tf.layers.batch_normalization(conv2)
                conv2=tf.nn.relu(conv2)
                conv2=af.x_unit(conv2)
                print(conv2)
            with tf.variable_scope('global_info'):
                f_shape = conv2.get_shape()
                feature = tf.layers.conv1d(conv2,self.class_num,(int(f_shape[1])),(1))
                feature = tf.layers.flatten(feature)
                print(feature)
        return feature


    def load(self, checkpoint_dir):
        print("Loading model ...")
        model_name = os.path.join(checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(model_name)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(model_name, ckpt_name))
            print("Load successful.")
            return True
        else:
            print("Load fail!!!")
            exit(0)

    def train(self,dataset,test_dataset):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        loss_list=[]
        oa_init=0
        for i in range(self.epoch):
            train_data,train_label = self.sess.run(dataset)
            # print(train_data.shape,train_label.shape)
            l,_,summery= self.sess.run([self.loss_total,self.optimizer,self.merged],feed_dict={self.image:train_data,self.label:train_label})
            if i % 1000 == 0:
                print(i,'step:',l)
                loss_list.append(l)
            if i % 5000 == 0:
                oa=self.test(test_dataset)
                if oa > oa_init:
                    oa_init=oa
                    self.saver.save(self.sess,os.path.join(self.model,self.model_name),global_step=i)
                    print('saved...')
            self.summary_write.add_summary(summery,i)
        loss_list=np.asarray(loss_list)
        sio.savemat(os.path.join(self.result,'loss.mat'),{'loss_list':loss_list})

    def test(self,dataset):
        test_dataset = dataset.data_parse(os.path.join(self.tfrecords, 'test_data.tfrecords'), type='test')
        acc_num,test_num = 0,0
        matrix = np.zeros((self.class_num,self.class_num),dtype=np.int64)
        try:
            while True:
                test_data, test_label = self.sess.run(test_dataset)
                pre_label = self.sess.run(self.pre_label, feed_dict={self.image:test_data,self.label:test_label})
                pre_label = np.argmax(pre_label,1)
                pre_label = np.expand_dims(pre_label,1)
                acc_num += np.sum((pre_label==test_label))
                test_num += test_label.shape[0]
                print(acc_num,test_num,acc_num/test_num)
                print(pre_label.shape[0])
                for i in range(pre_label.shape[0]):
                    matrix[pre_label[i],test_label[i]]+=1
        except tf.errors.OutOfRangeError:
            print("test end!")

        ac_list = []
        for i in range(len(matrix)):
            ac = matrix[i, i] / sum(matrix[:, i])
            ac_list.append(ac)
            print(i+1,'class:','(', matrix[i, i], '/', sum(matrix[:, i]), ')', ac)
        print('confusion matrix:')
        print(np.int_(matrix))
        print('total right num:', np.sum(np.trace(matrix)))
        accuracy = np.sum(np.trace(matrix)) / np.sum(matrix)
        print('oa:', accuracy)
        # kappa
        kk = 0
        for i in range(matrix.shape[0]):
            kk += np.sum(matrix[i]) * np.sum(matrix[:, i])
        pe = kk / (np.sum(matrix) * np.sum(matrix))
        pa = np.trace(matrix) / np.sum(matrix)
        kappa = (pa - pe) / (1 - pe)
        ac_list = np.asarray(ac_list)
        aa = np.mean(ac_list)
        oa = accuracy
        print('aa:',aa)
        print('kappa:', kappa)

        sio.savemat(os.path.join(self.result, 'result.mat'), {'oa': oa,'aa':aa,'kappa':kappa,'ac_list':ac_list,'matrix':matrix})
        return oa
    def save_decode_map(self,dataset):
        info = sio.loadmat(os.path.join(self.result,'info.mat'))
        data_gt = info['data_gt']
        # plt.figure(figsize=(map.shape[1] / 5, map.shape[0] / 5), dpi=100)# set size
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
        plt.axis('off')
        plt.pcolor(data_gt, cmap='jet')
        plt.savefig(os.path.join(self.result, 'groundtrouth.png'), format='png')
        plt.close()
        print('Groundtruth map get finished')
        de_map = np.zeros(data_gt.shape,dtype=np.int32)
        try:
            while True:
                map_data,pos = self.sess.run(dataset)
                pre_label = self.sess.run(self.pre_label, feed_dict={self.image:map_data})
                pre_label = np.argmax(pre_label,1)
                for i in range(pre_label.shape[0]):
                    [r,c]=pos[i]
                    de_map[r,c] = pre_label[i]
        except tf.errors.OutOfRangeError:
            print("test end!")
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
        plt.axis('off')
        plt.pcolor(de_map, cmap='jet')
        plt.savefig(os.path.join(self.result, 'decode_map.png'), format='png')
        plt.close()
        print('decode map get finished')