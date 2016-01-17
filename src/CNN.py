
import numpy as np
import scipy
import sys
import caffe
from math import log

caffe.set_mode_gpu()
caffe.set_device(0)
caffe_root = '/mnt/caffe-master/'


class CNN:
	
	def __init__(self, model, deploy, mean_value=np.asarray([104,  117,  123]), crop_size=227, batch_size=1, feature_blob='pool5',log="../log/cnn.log", Test = False):
		net = caffe.Classifier(deploy, model, caffe.TEST)

		transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
		transformer.set_transpose('data', (2,0,1))
		transformer.set_mean('data', mean_value) # mean pixel
		transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
		transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
		net.blobs['data'].reshape(batch_size, 3, crop_size, crop_size)
		self.net = net
		self.transformer = transformer
		self.feature_blob = feature_blob
		self.log_file = open(log,"w")
		self.Test = Test

	def get_class(self, image_path, prob=False):
		self.net.blobs['data'].data[0]= self.transformer.preprocess('data', caffe.io.load_image(image_path))
		out = self.net.forward()
		feat = self.net.blobs[self.feature_blob].data[0].reshape((9216,))
		class_index = out['prob'][0].argmax()
		class_prob = out['prob'][0]
		self.log_file.write("output class index: %d with probability: %f\n"%(class_index, class_prob[class_index]))
		return (class_index, class_prob[class_index])

if __name__ == "__main__":
	cnn = CNN("/mnt/caffenet.model", "/mnt/caffenet.deploy")
	print cnn.get_class("/mnt/data/r-cnn/VOC2007/JPEGImages/000067.jpg")

