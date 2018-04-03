import VAE as vaem
from mxnet import nd
import utils
import sys
#sys.path.append('..') Add the upper directory
sys.path.append('./dependencies')
import utils
ctx = utils.try_gpu()

filename = './params/vae.params.save'
vae = vaem.VAE()
vae.collect_params()
print(vae)
vae.load_params(filename, ctx = ctx)

# Show the first 3 images recovered by VAE
'''batch_size = 3
train_data, test_data = utils.load_data_mnist(batch_size)
for data, label in train_data:
	loss = vae(data)
	print(loss)
	utils.show_images_1D(vae.output)
	raw_input()'''

# Show the images generated from random noise by VAE
z = nd.random_normal(0,3,shape = (3,20))
x = vae.decode(z)
utils.show_images_1D(x)

