import argparse
import os
import tensorflow as tf

# from element_p.element.data_loader import Data
# from element_p.element.model import Model
# import element_p.element.unit as unit
from attention.data_loader import Data
from attention.model import Model
import attention.unit as unit


parser = argparse.ArgumentParser(description='Copresion and Classification for HSI')

parser.add_argument('--result',dest='result',default='result')
parser.add_argument('--log',dest='log',default='log')
parser.add_argument('--model',dest='model',default='model')
parser.add_argument('--tfrecords',dest='tfrecords',default='tfrecords')
parser.add_argument('--data_name',dest='data_name',default='Indian_pines')
parser.add_argument('--data_path',dest='data_path',default="./data")

parser.add_argument('--use_lr_decay',dest='use_lr_decay',default=True)
parser.add_argument('--decay_rete',dest='decay_rete',default=0.95)
parser.add_argument('--learning_rate',dest='lr',default=0.001)
parser.add_argument('--train_num',dest='train_num',default=150) # intger for number and decimal for percentage
parser.add_argument('--batch_size',dest='batch_size',default=200)
parser.add_argument('--fix_seed',dest='fix_seed',default=False)
parser.add_argument('--seed',dest='seed',default=666)
parser.add_argument('--decay_steps',dest='decay_steps',default=5000)
parser.add_argument('--test_batch',dest='test_batch',default=5000)
parser.add_argument('--epoch',dest='epoch',default=30001)
parser.add_argument('--save_decode_map',dest='save_decode_map',default=True)
parser.add_argument('--load_model',dest='load_model',default=True)


args = parser.parse_args()
if not os.path.exists(args.model):
    os.mkdir(args.model)
if not os.path.exists(args.log):
    os.mkdir(args.log)
if not os.path.exists(args.result):
    os.mkdir(args.result)
if not os.path.exists(args.tfrecords):
    os.mkdir(args.tfrecords)


def main():

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    num=10
    for i in range(num):
        args.id = str(i)
        tf.reset_default_graph()
        with tf.Session(config=config) as sess:
            args.result = os.path.join(args.result,args.id)
            args.log = os.path.join(args.log, args.id)
            args.model = os.path.join(args.model, args.id)
            args.tfrecords = os.path.join(args.tfrecords, args.id)
            if not os.path.exists(args.model):
                os.mkdir(args.model)
            if not os.path.exists(args.log):
                os.mkdir(args.log)
            if not os.path.exists(args.result):
                os.mkdir(args.result)
            if not os.path.exists(args.tfrecords):
                os.mkdir(args.tfrecords)

            dataset = Data(args)
            dataset.read_data()
            train_dataset = dataset.data_parse(os.path.join(args.tfrecords, 'train_data.tfrecords'), type='train')
            # test_dataset = dataset.data_parse(os.path.join(args.tfrecords, 'test_data.tfrecords'), type='test')
            map_dataset = dataset.data_parse(os.path.join(args.tfrecords, 'map_data.tfrecords'), type='map')

            model = Model(args,sess)
            if not args.load_model:
                model.train(train_dataset,dataset)
            else:
                model.load(args.model)
                model.test(dataset)
            if args.save_decode_map:
                model.save_decode_map(map_dataset)
            args.result = 'result'
            args.log = 'log'
            args.tfrecords = 'tfrecords'
            args.model = 'model'

    unit.average(args,num)



if __name__ == '__main__':
    main()