
from data import *
from q_network import DeepQLearner
import random
import cPickle
import sys

sys.setrecursionlimit(2000)

max_iters = 3010000
num_random_crop = 5
num_iter = 25
num_iter_other = 25
batch_size = 16
random_action_ratio = 0.1
train_path = "../input/trainval.list"
num_images = 5011
step = 500000
snapshot = 500000
sample_ratio = -0.01
class_index = -1
class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
				'bus', 'car', 'cat', 'chair', 'cow', 
				'diningtable', 'dog', 'horse', 'motorbike', 'person',
				'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
def train():
	qnetwork = DeepQLearner(84, 84, 17,
						6, .9, .00025, .95,
						 .01, 0.0, 0, 5000,
						16, 'nature_dnn', 'sgd',
						'sum', rng = np.random.RandomState(1234))
	#get Q(s,a) training set
	iters = 0
	class_index = int(sys.argv[1])
	min_lambda = float(sys.argv[2])
	small_scale = int(sys.argv[3])
	global sample_ratio
	fin = open("../input/"+class_names[class_index]+"_test.list", "r")
	t = len(fin.readlines())
	sample_ratio = t*0.1 / (5000-t)
	print "sample_ratio:%f"%(sample_ratio)
	IoU_count = np.ones(10, dtype=float)
	delta_lambda = 0.0
	data = Data(train_path, batch_size = batch_size, class_name = class_names[class_index])

#	qnetwork = cPickle.load(open("../output/qnetwork/qn_6.pkl","r"))
#	iters = 3010000
	terminals = np.zeros(batch_size, dtype=np.int32).reshape((batch_size, 1))
#	print "update counter: %d"%(qnetwork.update_counter)
	last_index = iters / 5000
	while iters < max_iters:
		data.init()
		EOF = 0
		while not EOF and iters < max_iters:
			EOF = data.next_image()
			while (len(data.cur_box) == 0) and (random.random() > sample_ratio) and not EOF:
				EOF = data.next_image()
			if not EOF:
				for k in range(num_random_crop):
					ith = random.randint(0, 33)
					if small_scale:
						ith = random.randint(18, 33)
					loop = True

#					_lambda = random.uniform(max(min_lambda, 0.9 - iters*1.5/max_iters * (1 - min_lambda)), 1.0)
#					_lambda = max(min_lambda, 1.0 - iters*1.5/max_iters * (1 - min_lambda))
					if (iters / 5000 == last_index + 1):
						if  (IoU_count[5:] / IoU_count.sum()).sum() < 0.20:
							delta_lambda = min(0.9, np.max(0.8 - IoU_count[5:]/IoU_count.sum()) + delta_lambda)
						else:
							delta_lambda = delta_lambda * 0.5
						print "IoU dis:" + str(IoU_count/IoU_count.sum())
						last_index = last_index + 1
						print "delta_lambda:%f"%(delta_lambda)
						IoU_count = np.ones(10, dtype=float)

					_lambda = min(0.9999999, max(min_lambda + delta_lambda, 1 - iters*1.5/max_iters * (1 - min_lambda)))

					data.random_crop(ith, _lambda)
					t=0
					last_action = 0
					tmp_iter = num_iter
					if len(data.cur_box) == 0:
						tmp_iter = num_iter_other
					stop_count = 0
					while (t!=tmp_iter) and loop:
						if ((iters % snapshot) == 0):
							qnetwork.save_q_network(iters/snapshot)
						state = data.get_state()
						action = -1

						random_action_ratio = max(0.15, 1.0 - iters * 1.5 / max_iters)
						if last_action == 16: #stop
							stop_count = stop_count + 1
						if random.random() < random_action_ratio:
							if random.random() < 0.1 / random_action_ratio:
								action = data.cal_best_action()
							else:
								action = random.randint(0, 16)
							qnetwork.random_action = True
						else:
							action = qnetwork.choose_action(state, 0.0)
						last_action = action
						data.update_train_db(t, state, action)			
						IoU_count[min(9, int(data.overlap_ratio *10))] = IoU_count[min(9, int(data.overlap_ratio *10))] + 1

						if not qnetwork.random_action:
							if len(data.cur_box) == 0:
								print "background iter: %d, overlap_ratio: %f, max_q_value: %f"%(iters, data.overlap_ratio, qnetwork.max_q_value)
							else:
								print "iter: %d, overlap_ratio: %f, max_q_value: %f"%(iters, data.overlap_ratio, qnetwork.max_q_value)
						#train QWork
						batch = data.draw_random_batch(batch_size)
						if type(batch) != type(-1):
							states, actions, rewards, next_states = batch
							for i in range(actions.shape[0]):
								if actions[i][0] == 16:
									terminals[i][0] = 1
								else:
									terminals[i][0] = 0
							loss = qnetwork.train(states,\
							 			actions, 
							 			rewards, 
							 			next_states, 
										terminals)
							print "iter: %d, loss: %f"%(iters, loss)
						data.next_crop(action)
						if (not data.not_outside()) or (stop_count == 2):
							loop = False
						#clone Q-network
						t = t + 1
						iters = iters + 1

if __name__=="__main__":
	train()
