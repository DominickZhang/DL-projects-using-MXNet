from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import nn
import mxnet as mx
import numpy as np
import os
from matplotlib import pyplot as plt

# Add utils.py etc.
import sys
sys.path.append('./dependencies')
import utils
ctx = utils.try_gpu()

#if not updating the seed by system time, you'll get the same results 
import time
seed = int(time.time()*100)
mx.random.seed(seed)

# Add DCGAN models
import DCGAN as dc

# Parameter Setting
epochs = 5
batch_size = 64
z_size = 100
lr = 0.0002
beta1 = 0.5
data_path = './dataset/lfw_dataset'

# Data Preprocess
img_list = []

def transform(data):
	data = mx.image.imresize(data, 64, 64) # state size: (64, 64, 3)
	data = nd.transpose(data, (2, 0, 1))
	data = data.astype(np.float32)/127.5 - 1 # normalize to [-1, 1]
	if data.shape[0] == 1:
		data = nd.tile(data, (3, 1, 1)) # if image is greyscale, repeat 3 times to get RGB image
	return data.reshape((1, ) + data.shape)

def visualize(img_arr):
	plt.imshow(((img_arr.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
	plt.axis('off')


for path, _, fnames in os.walk(data_path):
	for fname in fnames:
		if not fname.endswith('.jpg'):
			continue
		img = os.path.join(path, fname)
		img_arr = mx.image.imread(img)
		img_arr = transform(img_arr)
		img_list.append(img_arr)
train_data = mx.io.NDArrayIter(data = nd.concatenate(img_list), batch_size = batch_size)

'''for i in range(4):
	plt.subplot(1, 4, i+1)
	visualize(img_list[i + 10][0])
plt.show()'''

# Initial Model
	## loss
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
	## generator and discriminator
filename1 = './params/dcgan.netG.get'
filename2 = './params/dcgan.netD.get'
netG = dc.Generator()
netD = dc.Discriminator()
#netG.load_params(filename1, ctx = ctx)
#netD.load_params(filename2, ctx = ctx)
netG.initialize(mx.init.Normal(0.02), ctx = ctx)
netD.initialize(mx.init.Normal(0.02), ctx = ctx)
	## trainer for the generator and the discriminator
trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})

# Training Loop
from datetime import datetime
import logging

real_label = nd.ones((batch_size,), ctx=ctx)
fake_label = nd.zeros((batch_size,),ctx=ctx)

def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()
metric = mx.metric.CustomMetric(facc)

stamp =  datetime.now().strftime('%Y_%m_%d-%H_%M')
logging.basicConfig(level=logging.DEBUG)

for epoch in range(epochs):
    tic = time.time()
    btic = time.time()
    train_data.reset()
    iter = 0
    for batch in train_data:
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        data = batch.data[0].as_in_context(ctx)
        latent_z = mx.nd.random_normal(0, 1, shape=(batch_size, z_size, 1, 1), ctx=ctx)

        with autograd.record():
            # train with real image
            output = netD(data).reshape((-1, 1))
            #print(output.shape)
            #print(real_label.shape)
            errD_real = loss(output, real_label)
            metric.update([real_label,], [output,])

            # train with fake image
            fake = netG(latent_z)
            # fake.detach() is import here to avoid updating G when upadating D
            output = netD(fake.detach()).reshape((-1, 1))
            errD_fake = loss(output, fake_label)
            errD = errD_real + errD_fake
            errD.backward()
            metric.update([fake_label,], [output,])

        trainerD.step(batch.data[0].shape[0])

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        with autograd.record():
            fake = netG(latent_z)
            output = netD(fake).reshape((-1, 1))
            errG = loss(output, real_label)
            errG.backward()

        trainerG.step(batch.data[0].shape[0])

        # Print log infomation every ten batches
        if iter % 10 == 0:
            name, acc = metric.get()
            logging.info('speed: {} samples/s'.format(batch_size / (time.time() - btic)))
            logging.info('discriminator loss = %f, generator loss = %f, binary training acc = %f at iter %d epoch %d'
                     %(nd.mean(errD).asscalar(),
                       nd.mean(errG).asscalar(), acc, iter, epoch))
        iter = iter + 1
        btic = time.time()

    name, acc = metric.get()
    metric.reset()
    

filename3 = './params/dcgan.netG.save'
filename4 = './params/dcgan.netD.save'
netG.save_params(filename3)
netD.save_params(filename4)



