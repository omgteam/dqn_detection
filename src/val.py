from data import *
from q_network import DeepQLearner
import numpy as np
import sys
import cPickle
from CNN import *

num_random_crop = 34
num_iter = 20
total = 4952
IoU = 9.0
sIoU = 8.0
scoreThreshold = 0.0
gamma = 0.9
NMSFactor = 9.9
local_search_index_threshold = 3.0
class_index = -1
predict_index = -1
class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
				'bus', 'car', 'cat', 'chair', 'cow', 
				'diningtable', 'dog', 'horse', 'motorbike', 'person',
				'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
class Element:
	#v_value = sum_reward + V(s)
	def __init__(self, box, class_index, class_prob, overlap_ratio, q_val):
		self.box = box
		self.class_index = class_index
		self.class_prob = class_prob
		self.overlap_ratio = overlap_ratio
		self.local_search_index = 0.0
		self.q_val = q_val 
	def __str__(self):
		return str(self.box)+" " + str(self.overlap_ratio)+" " + str(self.local_search_index)+" " + str(self.IoU)

def NMS(past_list, current_box):
	for past_box in past_list:
		if get_overlapping_ratio(past_box, current_box) >= NMSFactor:
			return True
	return False

def kick_overlapped(sorted_list):
	loop = True
	index = 0
	if len(sorted_list) <= 1:
		loop = False
	while(loop):
		remove_list = []
		for k in range(index+1, len(sorted_list)):
			overlap_ratio = get_overlapping_ratio(sorted_list[index].box, sorted_list[k].box)
			if overlap_ratio > IoU:
				remove_list.insert(0,k)
			elif (overlap_ratio > sIoU) and (sorted_list[index].class_index == sorted_list[k].class_index):
				remove_list.insert(0,k)
		for k in remove_list:
			sorted_list.remove(sorted_list[k])
		index = index+1
		if(index > len(sorted_list)-2):
			loop = False
	return sorted_list

def local_search(box, cur_size):
	local_boxes = []
	local_boxes.append([(3*box[0] - box[2])/2, box[1], \
						(box[0] + box[2])/2, box[3]])

	local_boxes.append([(box[0] + box[2])/2, box[1], \
						(-box[0] + 3*box[2])/2, box[3]])

	local_boxes.append([box[0], (3*box[1] - box[3])/2, \
						box[2], (box[1] + box[3])/2])

	local_boxes.append([box[0], (box[1] + box[3])/2, \
						box[2], (-box[1] + 3*box[3])/2])

	return local_boxes

def transform_box(crop_box, cur_size):
	return [max(0, crop_box[0]), max(0, crop_box[1]), \
		min(cur_size[0], crop_box[2]), min(cur_size[1], crop_box[3])]

class TestSystem:
	def __init__(self, snapshot_index=0):
		global class_index
		global predict_index
		class_index = int(sys.argv[2])
		predict_index = int(sys.argv[3])

		self.data = Data("../input/"+class_names[predict_index]+"_test.list", log="../log/test.data", Test=True, class_name = class_names[class_index])
		self.local_search_log = open("../log/local_search.log","w")
		qnetwork = cPickle.load(open("../output/qnetwork/"+str(snapshot_index)+".pkl","r"))
		self.qnetwork=qnetwork
		self.qnetwork.reset_q()
		self.cnn = CNN("/mnt/caffenet.model", "/mnt/caffenet.deploy")
	
	def local_search_element(self, element, cur_size):
		local_boxes = local_search(element.box, cur_size)
		res_element = [element]
		for box in local_boxes:
			loop = True
			self.data.crop_box = box
			t = 5
			prob = 0.0
			overlap_ratio = 0.0
			tmp_element = -1
			while (t!=0) and loop:
				state = self.data.get_state()
				action = self.qnetwork.choose_action_test(state)
				overlap_ratio = get_overlapping_ratio(self.data.crop_box, self.data.cur_box)
				new_element = Element(self.data.crop_box,\
							class_index, 0, overlap_ratio, -1)
				crop_img = self.data.next_crop(action, False)
				if (not self.data.not_outside()) or (action==16) :
					loop = False
				t=t-1
				tmp_element = new_element
			res_element.append(tmp_element)
		size = len(res_element)
		co_overlap_mat = np.zeros(size*size, dtype=float).reshape((size, size))
		for i in range(size):
			for j in range(i, size):
				co_overlap_mat[i][j] = get_overlapping_ratio(res_element[i].box, res_element[j].box) / 10.0
				co_overlap_mat[j][i] = co_overlap_mat[i][j]
		for i in range(size):
			res_element[i].local_search_index = co_overlap_mat[i].sum()
		sorted_list = res_element
		for ele in sorted_list:
			self.local_search_log.write(str(ele) + "\n")
		self.local_search_log.write("\n")
		if sorted_list[0].local_search_index > local_search_index_threshold:
			return sorted_list[0]
		else:
			return -1

	def gen_initial_boxes(self, cur_size):
		res = []
		res.append([0, 0, cur_size[0], cur_size[1]])
		res.append([0,0,int(0.5*cur_size[0]), int(0.5*cur_size[1])])
		res.append([int(0.5*cur_size[0]),0,int(cur_size[0]), int(0.5*cur_size[1])])
		res.append([0,int(0.5*cur_size[1]),int(0.5*cur_size[0]), int(cur_size[1])])
		res.append([int(0.5*cur_size[0]),int(0.5*cur_size[1]),int(cur_size[0]), int(cur_size[1])])
		res.append([int(0.00*cur_size[0]),int(0.00*cur_size[1]),int(0.25*cur_size[0]), int(0.25*cur_size[1])])
		res.append([int(0.25*cur_size[0]),int(0.00*cur_size[1]),int(0.50*cur_size[0]), int(0.25*cur_size[1])])
		res.append([int(0.50*cur_size[0]),int(0.00*cur_size[1]),int(0.75*cur_size[0]), int(0.25*cur_size[1])])
		res.append([int(0.75*cur_size[0]),int(0.00*cur_size[1]),int(1.00*cur_size[0]), int(0.25*cur_size[1])])
		res.append([int(0.00*cur_size[0]),int(0.25*cur_size[1]),int(0.25*cur_size[0]), int(0.50*cur_size[1])])
		res.append([int(0.25*cur_size[0]),int(0.25*cur_size[1]),int(0.50*cur_size[0]), int(0.50*cur_size[1])])
		res.append([int(0.50*cur_size[0]),int(0.25*cur_size[1]),int(0.75*cur_size[0]), int(0.50*cur_size[1])])
		res.append([int(0.75*cur_size[0]),int(0.25*cur_size[1]),int(1.00*cur_size[0]), int(0.50*cur_size[1])])
		res.append([int(0.00*cur_size[0]),int(0.50*cur_size[1]),int(0.25*cur_size[0]), int(0.75*cur_size[1])])
		res.append([int(0.25*cur_size[0]),int(0.50*cur_size[1]),int(0.50*cur_size[0]), int(0.75*cur_size[1])])
		res.append([int(0.50*cur_size[0]),int(0.50*cur_size[1]),int(0.75*cur_size[0]), int(0.75*cur_size[1])])
		res.append([int(0.75*cur_size[0]),int(0.50*cur_size[1]),int(1.00*cur_size[0]), int(0.75*cur_size[1])])
		res.append([int(0.00*cur_size[0]),int(0.75*cur_size[1]),int(0.25*cur_size[0]), int(1.00*cur_size[1])])
		res.append([int(0.25*cur_size[0]),int(0.75*cur_size[1]),int(0.50*cur_size[0]), int(1.00*cur_size[1])])
		res.append([int(0.50*cur_size[0]),int(0.75*cur_size[1]),int(0.75*cur_size[0]), int(1.00*cur_size[1])])
		res.append([int(0.75*cur_size[0]),int(0.75*cur_size[1]),int(1.00*cur_size[0]), int(1.00*cur_size[1])])
		return res

	def test_xml_image(self):
		fout =[]
		for i in range(1, predict_index+1):
			fout.append(open("../output/test/comp3_det_test_"+class_names[i-1]+".txt","r"))
		fout.append(open("../output/test/comp3_det_test_"+class_names[predict_index]+".txt","w"))
		count = 0
		index = 0
		upper_count = 0
		while((not self.data.next_image()) and index != total):
			print self.data.xml_image
			imgid = self.data.xml_image.split("ns/")[1].split(".xml")[0]
			output_list = []
			upper_counted = False
			initial_boxes = self.gen_initial_boxes(self.data.cur_size)
			k = 0
			while len(initial_boxes) != 0:
				print "%dth crop"%(k)
				k = k + 1
				past_list = []
				loop = True
				self.data.crop_box = initial_boxes.pop()
				t = num_iter
				prob = 0.0
				overlap_ratio = 0.0
				tmp_element = -1
				while (t!=0) and loop:
					state = self.data.get_state()
					action = self.qnetwork.choose_action_test(state)
					overlap_ratio = get_overlapping_ratio(self.data.crop_box, self.data.cur_box)
					if overlap_ratio > 5 and (not upper_counted):
						upper_count = upper_count + 1
						upper_counted = True
					new_element = Element(transform_box(self.data.crop_box, self.data.cur_size),\
								predict_index, -1, overlap_ratio, self.qnetwork.max_q_value)
					print "%f, %s, %f"%(overlap_ratio, str(self.data.crop_box), self.qnetwork.max_q_value)
					self.data.next_crop(action, False)

					if (not self.data.not_outside()) or (action==16) :
						loop = False
					if NMS(past_list, self.data.crop_box):
						loop = False
					else:
						past_list.append(self.data.crop_box)
					if tmp_element == -1 or tmp_element.q_val > new_element.q_val:
						tmp_element = new_element
					t=t-1
				if tmp_element != -1:
					self.data.save_crop_image(tmp_element.box, "../log/test.jpg")
					class_predict, proba = self.cnn.get_class("../log/test.jpg")
					print "%d:%f"%(class_predict, proba)
					if class_predict == class_index + 1:				
						tmp_element.local_search_index = proba
						print "end: %f, %s, %f"%(tmp_element.overlap_ratio, str(tmp_element.box), tmp_element.local_search_index)
						output_list.append(tmp_element)
					else:
						print "-1"
				else:
					print "-1"
			sorted_list = sorted(output_list, key = lambda element: element.local_search_index, reverse=True)

			res = False
			for ele in sorted_list:
				if ele.overlap_ratio > 5.0:
					res = True
				fout[ele.class_index].write(imgid+" "+str(ele.local_search_index)+\
					" "+str(ele.box[0])+" "+str(ele.box[1])+" "+str(ele.box[2])+" "+str(ele.box[3]) +"\n")
			if res:
				count = count + 1
				print "Hitted"
			else:
				print "Missed"
			index = index + 1
		print "Hitted count, Upper Count,  Total: %d, %d, %d"%(count, upper_count, total)

def test():
	snapshot_index = 0
	if len(sys.argv) != 1:
		snapshot_index = sys.argv[1]
	testSystem = TestSystem(snapshot_index)
	testSystem.test_xml_image()

if __name__ == "__main__":
	test()
