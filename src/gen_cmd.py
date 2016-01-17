#tmp = [2,4,6,8,11,14,15]
#tmp = [2,4,6,8]
for k in range(20):
	if k % 2:
		print "cp -r /mnt/DQN /mnt/DQN_%d && cd /mnt/DQN_%d/src && THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python train.py %d 0.0 0  1>> ../log/train 2>> ../log/train &"%(k, k, k)
	else:
		print "cp -r /mnt/DQN /mnt/DQN_%d && cd /mnt/DQN_%d/src && THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python train.py %d 0.0 0  1>> ../log/train 2>> ../log/train "%(k, k, k)

"""
	k = tmp[i]
	print "cd /mnt/DQN_%d/src && python processing_log.py > ../output/analysis.txt "%(k)


for i in range(4):
	print "cp -r /mnt/DQN/src/processing_log.py /mnt/DQN_%d/src/"%(i)
"""
