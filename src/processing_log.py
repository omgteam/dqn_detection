from matplotlib import pyplot as plt
import numpy as np
import random
import re
from math import sqrt

def L2():
	fin = open("../log/qn.log", "r")
	line = fin.readline()
	while line!= "":
		if line.find("L2")!=-1:
			print line.strip().split(":")[1]
		line = fin.readline()

def draw():
	Y = np.loadtxt("../log/L2", dtype=float)
	X = range(1, len(Y)+1)
	plt.plot(X,Y)
	plt.savefig("../output/L2.jpg")

def batch_cost():
	fin = open("../../QN_VOC2007v4/log/qn.log", "r")
	line = fin.readline()
	res = []
	while line!= "":
		if line.find("batch cost")!=-1 and random.random() < 0.1:
			res.append(float(line.strip().split(":")[1]))
		line = fin.readline()
	Y = np.asarray(res)
	X = range(1, len(Y)+1)
	plt.figure()
	plt.plot(X,Y, '.')
	plt.axis((0, len(X), 0, 100))
	plt.savefig("../output/bc4.jpg")

def L2_cost():
	fin = open("../../QN_VOC2007v4/log/qn.log", "r")
	line = fin.readline()
	res = []
	while line!= "":
		if line.find("L2")!=-1:
			res.append(float(line.strip().split(":")[1]))
		line = fin.readline()
	Y = np.asarray(res)
	X = range(1, len(Y)+1)
	plt.figure()
	plt.plot(X,Y,'.')
	plt.axis((0, len(X), 0, 100))
	plt.savefig("../output/l2_4.jpg")

def test():
	batch_cost()
	L2_cost() 

def get_max(tmp_lines):
	nb_lines = []
	for line in tmp_lines:
		if line.split(",")[5] != " 0":
			nb_lines.append(line)
	if len(nb_lines) != 0:
		return sorted(nb_lines, key = lambda line: float(line.strip().split(",")[6]), reverse=True)[0]
	else:
		return sorted(tmp_lines, key = lambda line: float(line.strip().split(",")[6]), reverse=False)[0]
	
def stat(path):
	fin = open(path, "r")
	line = fin.readline()
	pattern = re.compile("[0-9]+th crop\n")
	start_IoUs = []
	end_IoUs = []
	while line != "":
		if pattern.match(line) != None:
			line = fin.readline().strip()
			tmp_lines = []
			while len(line.split(",")) == 5 or line == "-1":
				tmp_lines.append(line)
				line = fin.readline().strip()
#			if len(tmp_lines)!=0 and (tmp_lines[len(tmp_lines)-1].split(",")[5] != " 0"):
			if len(tmp_lines)!=0 and tmp_lines[len(tmp_lines)-1] != "-1":
				start_IoUs.append(float(tmp_lines[0].split(",")[0]))
				end_IoUs.append(float(tmp_lines[len(tmp_lines)-1].split(",")[0]))

		else:
			line = fin.readline()
	start_IoUs = np.asarray(start_IoUs)
	end_IoUs = np.asarray(end_IoUs)
	np.savetxt("../output/start_IoUs", start_IoUs)
	np.savetxt("../output/end_IoUs", end_IoUs)

def handle_local_search(thres = 3.5):
	fin = open("../log/local_search.log4","r")
	res = []
	line = fin.readline()
	while line!="":
		if line != "\n":
			splits = line.strip().split("] ")[1].split(" ")
			overlap_ratio = float(splits[0])
			score = float(splits[1])
			res.append([overlap_ratio, score])
		line = fin.readline()
	fp = 0
	tp = 0
	fn = 0
	tn = 0
	for ele in res:
		if ele[1] < thres:
			if ele[0] > 5.0:
				tn = tn + 1
			else:
				fn = fn + 1
		else:
			if ele[0] > 5.0:
				tp = tp + 1
			else:
				fp = fp + 1
	print "thres:%f fp:%d, tp:%d, fn:%d, tn:%d, acc:%f"%(thres, fp, tp, fn, tn, 1.0*tp/ (tp+fp))

def handle_train_log(path="../log/train"):
	fin = open(path, "r")
	line = fin.readline()
	IoU_p = 0.0
	iters = 0
	IoU_g = 0.0
	max_q = 0.0
	res = []
	res_b = []
	while line != "":
		if line.find("q_vals") != -1:
			line = fin.readline()
			splits = line.strip().split(", ")
			iters = int(splits[0].split(": ")[1])
			IoU_g = float(splits[1].split(": ")[1])
			max_q = float(splits[2].split(": ")[1])
			IoU_p = IoU_g
			if line.find("background")!=-1:
				res_b.append([iters, IoU_p, IoU_g, max_q])
			else:
				res.append([iters, IoU_p, IoU_g, max_q])
			line = fin.readline()
		else:
			line = fin.readline()
	res = np.asarray(res)
	np.savetxt("../output/train_res.txt", res, fmt='%10.4f')
	res_b = np.asarray(res_b)
	np.savetxt("../output/train_res_b.txt", res_b, fmt='%10.4f')
	
def analysis_IoU():
	res = np.loadtxt("../output/train_res.txt", dtype=float)
	window = 100000
	print "\nIoU"
	for i in range(res.shape[0]/window):
		data_p = res[i*window:i*window+window,1]
		data_g = res[i*window:i*window+window,2]
		diff = abs(data_p - data_g)
		print "%2.2f %2.2f"%(diff.mean(), diff.std())

def analysis_Q():
	res = np.loadtxt("../output/train_res.txt", dtype=float)
	window = 100000
	print res.shape
	stat = []
	for i in range(res.shape[0]/window):
		tmp = []
		for j in range(11):
			tmp.append([])
		for j in range(i*window, min(i*window + window, res.shape[0])):
			tmp[int(res[j][2]*10)].append(res[j][3])
		stat.append(tmp)
	print "\nQ_dis"
	for tmp in stat:
		line = ""
		for j in range(10):
			data = np.asarray(tmp[j])
			line = line + "%2.2f:%2.2f "%(data.mean(), data.std())
		print line

	print "\nQ_count"
	for tmp in stat:
		line = ""
		for j in range(10):
			data = np.asarray(tmp[j])
			line = line + str(len(tmp[j]))+" "
		print line

def analysis_IoU_b():
	res = np.loadtxt("../output/train_res_b.txt", dtype=float)
	window = 100000
	print "\nIoU_b"
	for i in range(res.shape[0]/window):
		data_p = res[i*window:i*window+window,1]
		data_g = res[i*window:i*window+window,2]
		diff = abs(data_p - data_g)
		print "%2.2f %2.2f"%(diff.mean(), diff.std())

def analysis_Q_b():
	res = np.loadtxt("../output/train_res_b.txt", dtype=float)
	window = 100000
	print res.shape
	stat = []
	for i in range(res.shape[0]/window):
		tmp = []
		for j in range(11):
			tmp.append([])
		for j in range(i*window, min(i*window + window, res.shape[0])):
			tmp[int(res[j][1]*10)].append(res[j][3])
		stat.append(tmp)
	print "\nQ_dis background"
	for tmp in stat:
		line = ""
		for j in range(10):
			data = np.asarray(tmp[j])
			line = line + "%2.2f:%2.2f "%(data.mean(), data.std())
		print line

	print "\nQ_count background"
	stat = []
	for i in range(res.shape[0]/window):
		tmp = []
		for j in range(11):
			tmp.append([])
		for j in range(i*window, min(i*window + window, res.shape[0])):
			tmp[int(res[j][1]*10)].append(res[j][3])
		stat.append(tmp)

	for tmp in stat:
		line = ""
		for j in range(10):
			data = np.asarray(tmp[j])
			line = line + str(len(tmp[j]))+" "
		print line

def analysis_total(path):
	fin = open(path, "r")
	lines = fin.readlines()
	print path
	for i in range(len(lines)):
		line = lines[i].strip()
		if line == "Q_count":
			print lines[i-2].strip()
		if line == "IoU":
			print lines[i-2].strip()
		if line == "Q_dis background":
			print lines[i-3].strip()
		if line == "IoU_b":
			print lines[i-2].strip()
	print lines[len(lines)-1].strip()
	print ""

def analysis_test_log(path, zero = False):
	f = open(path, "r")
	line = f.readline()
	a = []
	b = []
	while line != "":
#		print line
		splits = line.strip().split(" ")
		if (len(splits) == 6):	
			if zero:
				a.append(0.0)
			else:
				a.append(float(splits[0][0:len(splits[0])-2])/10)
			b.append(float(splits[5]))
		line = f.readline()

	a = np.asarray(a)
	b = np.asarray(b)
	r = (a - b)
	print len(r)
	print r.mean()
	print r.std()

def test2():
	start_IoUs = np.loadtxt("../output/start_IoUs") / 10
	end_IoUs = np.loadtxt("../output/end_IoUs") / 10

	assert(len(start_IoUs) == len(end_IoUs))
	print len(start_IoUs)
	plt.figure()
	plt.hist(start_IoUs, 20)
	plt.savefig("../output/hist_1.jpg")
	plt.figure()
	plt.hist(end_IoUs, 20)
	plt.savefig("../output/hist_2.jpg")
	plt.figure()

def test3():
	start_IoUs = np.loadtxt("../output/start_IoUs") / 10
	end_IoUs = np.loadtxt("../output/end_IoUs") / 10

	print start_IoUs.mean()

	print end_IoUs.mean()
	print np.histogram(end_IoUs, 10, normed=True, density=True)
	print "%fpercent"%(((end_IoUs >= 0.5).sum() * 1.0/ (end_IoUs >= 0).sum()))
			
def test5():
	stat("../log/test_3.log")
	test3()
	stat("../log/test_4.log")
	test3()

def test6():
	for i in range(20):
		analysis_total("/mnt/DQN_%d/output/analysis.txt"%(i))

def test7():
	handle_train_log("../log/train")
	analysis_Q()
	analysis_IoU()
	analysis_Q_b()
	analysis_IoU_b()


if __name__ == "__main__":
	test7()

#	analysis_test_log("../log/test_3.log")
#	analysis_test_log("../log/test_4.log")
#	analysis_test_log("../log/test_5.log")

"""
	analysis_test_log("../log/test_1.log")
	analysis_test_log("../log/test_2.log")
	analysis_test_log("../log/test_3.log")
	analysis_test_log("../log/test_3.log")
	analysis_test_log("../log/test_4.log")
	analysis_test_log("../log/test_5.log")
	analysis_test_log("../log/test_6.log")
	analysis_test_log("../log/test_7.log")
	analysis_test_log("../log/test_8.log")
#	analysis_test_log("/mnt/DQN_0/log/test_4.log")
#	analysis_test_log("/mnt/DQN_0/log/test_5.log")
"""

