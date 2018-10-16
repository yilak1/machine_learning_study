import mxnet as mx
import numpy as np
import logging


'''data is a list of NDArray, each of them has  nn  length first dimension. 
        For example, if an example is an image with size  224×224224×224  and RGB channels, 
        then the array shape should be (n, 3, 224, 244). Note that the image batch format used by MXNet is
        batch_size * num_channel * height * width
        The channels are often in RGB order
        Each array will be copied into a free variable of the Symbol later. 
        The mapping from arrays to free variables should be given by the provide_data attribute of the iterator, 
        which will be discussed shortly.
   label is also a list of NDArray. Often each NDArray is a 1-dimensional array with shape (n,). 
        For classification, each class is represented by an integer starting from 0.
   pad is an integer shows how many examples are for merely used for padding, which should be ignored 
       in the results. A nonzero padding is often used when we reach the end of the data and the total number 
       of examples cannot be divided by the batch size.
'''


class SimpleBatch(object):
    def __init__(self, data, label, pad=None):
        self.data = data
        self.label = label
        self.pad = pad


# Symbol and Data Variable

'''weight : the weight parameters
   bias : the bias parameters
   output : the output
   label : input label

'''
num_classes = 10
net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=64)
net = mx.sym.Activation(data=net, name='relu1', act_type='relu')
net = mx.sym.FullyConnected(data=net, name='fc2', num_hidden=num_classes)
net = mx.sym.SoftmaxOutput(data=net, name='softmax')
print(net.list_arguments())
print(net.list_outputs())

'''The following example define a matrix factorization object function with rank 10 for recommendation systems.
   It has three input variables, user for user IDs, item for item IDs, and score is the rating user 
   gives to item.
'''
num_users = 1000
num_items = 1000
k = 10
user = mx.symbol.Variable('user')
item = mx.symbol.Variable('item')
score = mx.symbol.Variable('score')
# user feature lookup
user = mx.symbol.Embedding(data = user, input_dim = num_users, output_dim = k)
# item feature lookup
item = mx.symbol.Embedding(data = item, input_dim = num_items, output_dim = k)
# predict by the inner product, which is elementwise product and then sum
pred = user * item
pred = mx.symbol.sum_axis(data = pred, axis = 1)
pred = mx.symbol.Flatten(data = pred)
# loss layer
pred = mx.symbol.LinearRegressionOutput(data = pred, label = score)


# data Iterators
class SimpleIter:
    def __init__(self, data_names, data_shapes, data_gen,
                 label_names, label_shapes, label_gen, num_batches=10):
        self._provide_data = zip(data_names, data_shapes)
        self._provide_label = zip(label_names, label_shapes)
        self.num_batches = num_batches
        self.data_gen = data_gen
        self.label_gen = label_gen
        self.cur_batch = 0

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        if self.cur_batch < self.num_batches:
            self.cur_batch += 1
            data = [mx.nd.array(g(d[1])) for d, g in zip(self._provide_data, self.data_gen)]
            assert len(data) > 0, "Empty batch data."
            label = [mx.nd.array(g(d[1])) for d, g in zip(self._provide_label, self.label_gen)]
            assert len(label) > 0, "Empty batch label."
            return SimpleBatch(data, label)
        else:
            raise StopIteration


# @@@ AUTOTEST_OUTPUT_IGNORED_CELL
logging.basicConfig(level=logging.INFO)

n = 32
data1 = SimpleIter(['data'], [(n, 100)],
                  [lambda s: np.random.uniform(-1, 1, s)],
                  ['softmax_label'], [(n,)],
                  [lambda s: np.random.randint(0, num_classes, s)])

mod = mx.mod.Module(symbol=net)
mod.fit(data1, num_epoch=5)
