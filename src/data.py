"""Image data store and interfaces"""

from PIL import Image
import numpy as np
import xml.dom.minidom
import random
from math import pow

num_classes = 21
step_cost = 0.0
def get_value_by_tag(xml, tag):
	return int(float(xml.getElementsByTagName(tag)[0].childNodes[0].data))

def get_name_by_tag(xml, tag):
	return xml.getElementsByTagName(tag)[0].childNodes[0].data

def cal_box_area(box):
	return (box[3]-box[1])*(box[2]-box[0])

def get_overlapping_area(box1, box2):
	xmin = max(box1[0], box2[0])
	ymin = max(box1[1], box2[1])
	xmax = min(box1[2], box2[2])
	ymax = min(box1[3], box2[3])
	if (xmin < xmax) and (ymin<ymax):
		return (ymax-ymin)*(xmax-xmin)
	else:
		return 0

def curriculum_tuning(gbox, crop_box, _lambda):
	gw = gbox[2] - gbox[0]
	gh = gbox[3] - gbox[1]
	gx = (gbox[2] + gbox[0]) / 2
	gy = (gbox[3] + gbox[1]) / 2
	crop_w = crop_box[2] - crop_box[0]
	crop_h = crop_box[3] - crop_box[1]
	crop_x = (crop_box[2] + crop_box[0]) / 2
	crop_y = (crop_box[3] + crop_box[1]) / 2
	tx = int(_lambda * gx + (1 - _lambda) * crop_x)
	ty = int(_lambda * gy + (1 - _lambda) * crop_y)
	tw = int(_lambda * gw + (1 - _lambda) * crop_w)
	th = int(_lambda * gh + (1 - _lambda) * crop_h)
	return [tx - tw/2, ty - th/2, tx + tw/2, ty + th/2]

def get_overlapping_ratio(box1, box2, contain=False):
	if len(box2)==0:
		return 0.0
	if (type(box2[0]) == type(1)):
		xmin = max(box1[0], box2[0])
		ymin = max(box1[1], box2[1])
		xmax = min(box1[2], box2[2])
		ymax = min(box1[3], box2[3])
		if (xmin < xmax) and (ymin<ymax):
			area =(1.0 * (ymax-ymin)*(xmax-xmin))
			if contain:
				return max(area * 10 / (cal_box_area(box1)), area * 10 / (cal_box_area(box2)))
			else:
				return area * 10 / (cal_box_area(box1) + cal_box_area(box2) - area)
		else:
			return 0.0
	else:
		res = 0.0
		for box in box2:
			res = max(res, get_overlapping_ratio(box1, box))
		return res

class Data:
	cur_image = ""
	xml_image = ""
	cur_box = [0,0,0,0]
	cur_size = [0,0]
	crop_box = [0,0,0,0]
	len_features = 0
	tmaxnum = 0
	states = []
	next_states = []
	actions = []
	rewards = []
	tindex = 0
	tfull = False
	log_file = ""
	action_definition=["Up 10%", "Up 30%", "Down 10%", "Down 30%", "Left 10%", "Left 30%", "Right 10%", "Right 30%",\
						"Taller 10%", "Taller 30%", "Shorter 10%", "Shorter 30%", "Wider 10%", "Wider 30%", "Thinner 10%", "Thinner 30%", "Stop"]
	def __init__(self, xml_path, D=80000, len_features=84*84*6, log="../log/data.log", D_threshold = 10000, Test=False, batch_size = 16, class_name=""):
		self.len_features = len_features
		self.tmaxnum = D
		self.xml_file = open(xml_path, "r")
		self.Test = Test
		self.overlap_ratio = 0.0
		self.class_name = class_name
		if not Test:
			self.states = np.zeros(D*84*84*6, dtype=np.int16).reshape((D, 6, 84,84))
			self.masks = np.zeros(D, dtype=np.int16)
			self.actions = np.zeros(D*1, dtype=np.int32).reshape((D, 1))
			self.rewards = np.zeros(D*1, dtype=np.float32).reshape((D, 1))

			self.batch_states = np.zeros(batch_size*84*84*6, dtype=np.float32).reshape((batch_size, 6, 84,84))
			self.batch_next_states = np.zeros(batch_size*84*84*6, dtype=np.float32).reshape((batch_size, 6, 84,84))
			self.batch_actions = np.zeros(batch_size*1, dtype=np.int32).reshape((batch_size, 1))
			self.batch_rewards = np.zeros(batch_size*1, dtype=np.float32).reshape((batch_size, 1))

		self.tfull = False
		self.tindex = 0
		self.log_file = open(log, "w")
		self.log_file.write("input: "+xml_path+"\n")
		if Test:
			self.crop_img = "../store/test_crop_img2.JPEG"
		else:
			self.D_threshold = D_threshold


	def init(self):
		self.xml_file.seek(0)
		self.log_file.write("new epoch\n")

	def not_outside(self):
		if get_overlapping_ratio(self.crop_box, [0,0,self.cur_size[0], self.cur_size[1]]) == 0.0:
			return False
		elif (abs(self.crop_box[0]-self.crop_box[2]) <= 10) or (abs(self.crop_box[1]-self.crop_box[3]) <= 10):
			return False
		elif (abs(self.crop_box[0]-self.crop_box[2]) > 2000) or (abs(self.crop_box[1]-self.crop_box[3]) > 2000):
			return False
		else:
			return True

	def random_crop(self, k, _lambda = 0.0):
		cur_size = self.cur_size
		crop_box = [0,0,0,0]
		if not self.Test:
			if k == 0:
				crop_box = [0,0, cur_size[0], cur_size[1]]
			else:
				scale = 0
				if k < 5:
					scale = 0.33
				elif k < 14:
					scale = 0.67
				elif k < 18:
					scale = 0.5
				else:
					scale = 0.75
				x = random.randint(0, int(scale*cur_size[0]))
				w = cur_size[0] - int(scale*cur_size[0])
				y = random.randint(0, int(scale*cur_size[1]))
				h = cur_size[1] - int(scale*cur_size[1])
				crop_box = [x, y, x+w, y+h]	
			if (len(self.cur_box) != 0):
				gbox = self.cur_box[random.randint(0, len(self.cur_box)-1)]
				crop_box = curriculum_tuning(gbox, crop_box, _lambda)
			else:
				gbox = self.other_box[random.randint(0, len(self.other_box)-1)]
				crop_box = curriculum_tuning(gbox, crop_box, _lambda)
		else:	
			if k == 0:
				crop_box = [0,0, cur_size[0], cur_size[1]]
			elif k == 1:
				crop_box = [0,0,int(0.67*cur_size[0]), int(0.67*cur_size[1])]
			elif k == 2:
				crop_box = [int(0.33*cur_size[0]),0,int(cur_size[0]), int(0.67*cur_size[1])]
			elif k == 3:
				crop_box = [0,int(0.33*cur_size[1]),int(0.67*cur_size[0]), int(cur_size[1])]
			elif k == 4:
				crop_box = [int(0.33*cur_size[0]),int(0.33*cur_size[1]),int(cur_size[0]), int(cur_size[1])]
			elif k == 14:
				crop_box = [0,0,int(0.5*cur_size[0]), int(0.5*cur_size[1])]
			elif k == 15:
				crop_box = [int(0.5*cur_size[0]),0,int(cur_size[0]), int(0.5*cur_size[1])]
			elif k == 16:
				crop_box = [0,int(0.5*cur_size[1]),int(0.5*cur_size[0]), int(cur_size[1])]
			elif k == 17:
				crop_box = [int(0.5*cur_size[0]),int(0.5*cur_size[1]),int(cur_size[0]), int(cur_size[1])]
			elif k == 5:
				crop_box = [0,0,int(0.33*cur_size[0]), int(0.33*cur_size[1])]
			elif k == 6:
				crop_box = [int(0.33*cur_size[0]),0,int(0.67*cur_size[0]), int(0.33*cur_size[1])]
			elif k == 7:
				crop_box = [int(0.67*cur_size[0]),0,int(cur_size[0]), int(0.33*cur_size[1])]
			elif k == 8:
				crop_box = [0,int(0.33*cur_size[1]),int(0.33*cur_size[0]), int(0.67*cur_size[1])]
			elif k == 9:
				crop_box = [int(0.33*cur_size[0]),int(0.33*cur_size[1]),int(0.67*cur_size[0]), int(0.67*cur_size[1])]
			elif k == 10:
				crop_box = [int(0.67*cur_size[0]),int(0.33*cur_size[1]),int(cur_size[0]), int(0.67*cur_size[1])]
			elif k == 11:
				crop_box = [0,int(0.67*cur_size[1]),int(0.33*cur_size[0]), int(cur_size[1])]
			elif k == 12:
				crop_box = [int(0.33*cur_size[0]),int(0.67*cur_size[1]),int(0.67*cur_size[0]), int(cur_size[1])]
			elif k == 13:
				crop_box = [int(0.67*cur_size[0]),int(0.67*cur_size[1]),int(cur_size[0]), int(cur_size[1])]
			elif k == 18:
				crop_box = [int(0.00*cur_size[0]),int(0.00*cur_size[1]),int(0.25*cur_size[0]), int(0.25*cur_size[1])]
			elif k == 19:
				crop_box = [int(0.25*cur_size[0]),int(0.00*cur_size[1]),int(0.50*cur_size[0]), int(0.25*cur_size[1])]
			elif k == 20:
				crop_box = [int(0.50*cur_size[0]),int(0.00*cur_size[1]),int(0.75*cur_size[0]), int(0.25*cur_size[1])]
			elif k == 21:
				crop_box = [int(0.75*cur_size[0]),int(0.00*cur_size[1]),int(1.00*cur_size[0]), int(0.25*cur_size[1])]
			elif k == 22:
				crop_box = [int(0.00*cur_size[0]),int(0.25*cur_size[1]),int(0.25*cur_size[0]), int(0.50*cur_size[1])]
			elif k == 23:
				crop_box = [int(0.25*cur_size[0]),int(0.25*cur_size[1]),int(0.50*cur_size[0]), int(0.50*cur_size[1])]
			elif k == 24:
				crop_box = [int(0.50*cur_size[0]),int(0.25*cur_size[1]),int(0.75*cur_size[0]), int(0.50*cur_size[1])]
			elif k == 25:
				crop_box = [int(0.75*cur_size[0]),int(0.25*cur_size[1]),int(1.00*cur_size[0]), int(0.50*cur_size[1])]
			elif k == 26:
				crop_box = [int(0.00*cur_size[0]),int(0.50*cur_size[1]),int(0.25*cur_size[0]), int(0.75*cur_size[1])]
			elif k == 27:
				crop_box = [int(0.25*cur_size[0]),int(0.50*cur_size[1]),int(0.50*cur_size[0]), int(0.75*cur_size[1])]
			elif k == 28:
				crop_box = [int(0.50*cur_size[0]),int(0.50*cur_size[1]),int(0.75*cur_size[0]), int(0.75*cur_size[1])]
			elif k == 29:
				crop_box = [int(0.75*cur_size[0]),int(0.50*cur_size[1]),int(1.00*cur_size[0]), int(0.75*cur_size[1])]
			elif k == 30:
				crop_box = [int(0.00*cur_size[0]),int(0.75*cur_size[1]),int(0.25*cur_size[0]), int(1.00*cur_size[1])]
			elif k == 31:
				crop_box = [int(0.25*cur_size[0]),int(0.75*cur_size[1]),int(0.50*cur_size[0]), int(1.00*cur_size[1])]
			elif k == 32:
				crop_box = [int(0.50*cur_size[0]),int(0.75*cur_size[1]),int(0.75*cur_size[0]), int(1.00*cur_size[1])]
			elif k == 33:
				crop_box = [int(0.75*cur_size[0]),int(0.75*cur_size[1]),int(1.00*cur_size[0]), int(1.00*cur_size[1])]

		self.crop_box = crop_box
		self.log_file.write("%dth random_crop, crop_box:%s\n"%(k, str(crop_box)))
		return 0

	def get_state(self):
		def extend_box(crop_box):
			w = crop_box[2] - crop_box[0]
			h = crop_box[3] - crop_box[1]
			return [crop_box[0] - w/4, crop_box[1] - h/4, crop_box[2] + w/4, crop_box[3] + h/4]
		img_data = np.array(self.cur_image.crop(self.crop_box).resize((84,84)))
		img_data_extend = np.array(self.cur_image.crop(extend_box(self.crop_box)).resize((84,84)))
		return np.concatenate((np.swapaxes(np.swapaxes(img_data, 0, 2), 1, 2), np.swapaxes(np.swapaxes(img_data_extend, 0, 2), 1, 2)))

	def next_crop(self, action, test=False):
		self.crop_box = self.cal_actioned_crop(action)
		if (test):
			self.crop_box[0] = max(0, self.crop_box[0])
			self.crop_box[1] = max(0, self.crop_box[1])
			self.crop_box[2] = min(self.cur_size[0], self.crop_box[2])
			self.crop_box[3] = min(self.cur_size[1], self.crop_box[3])

		if (self.not_outside()):
			return 0
		else:
			return -1

	def cal_best_action(self):
		overlaps = np.zeros(16, dtype=float)
		for i in range(16):
			actioned_crop_box = self.cal_actioned_crop(i)
			overlaps[i] = get_overlapping_ratio(actioned_crop_box, self.cur_box)
		if overlaps.max() > get_overlapping_ratio(self.crop_box, self.cur_box):
			return overlaps.argmax()
		else:
			return 16

	def cal_actioned_crop(self, action):
		"Up, Down, Left, Right, Wider, Thinner, Taller, Shorter * 0.3/0.5 or 0.2/0.4"
		y_len = self.crop_box[3] - self.crop_box[1]
		x_len = self.crop_box[2] - self.crop_box[0]
		y = (self.crop_box[3] + self.crop_box[1]) / 2
		x = (self.crop_box[2] + self.crop_box[0]) / 2

		if action == 0: #Up 0.15
			y = y + int(0.1  * y_len)
		elif action == 1: #Up 0.5
			y = y + int(0.5  * y_len)
		elif action == 2: #Down 0.15
			y = y - int(0.1  * y_len)
		elif action == 3: #Down 0.5
			y = y - int(0.5  * y_len)
		elif action == 4: #Left 0.15
			x = x - int(0.1  * x_len)
		elif action == 5: #Left 0.5
			x = x - int(0.5  * x_len)
		elif action == 6: #Right 0.15
			x = x + int(0.1  * x_len)
		elif action == 7: #Right 0.5
			x = x + int(0.5  * x_len)
		elif action == 8: #Taller 0.15
			y_len = y_len + int(0.1  * y_len)
		elif action == 9: #Taller 0.5
			y_len = y_len + int(0.5  * y_len)
		elif action == 10: #Shorter 0.15
			y_len = y_len - int(0.1  * y_len)
		elif action == 11: #Shorter 0.5
			y_len = y_len - int(0.5  * y_len)
		elif action == 12: #Widder 0.15
			x_len = x_len + int(0.1  * x_len)
		elif action == 13: #Widder 0.5
			x_len = x_len + int(0.5  * x_len)
		elif action == 14: #Thinner 0.15
			x_len = x_len - int(0.1  * x_len)
		elif action == 15: #Thinner 0.5
			x_len = x_len - int(0.5  * x_len)
		elif action == 16: #stop
			pass

		actioned_crop_box = [0,0,0,0]
		actioned_crop_box[0] = x-(x_len/2)
		actioned_crop_box[1] = y-(y_len/2)
		actioned_crop_box[2] = x+(x_len/2)
		actioned_crop_box[3] = y+(y_len/2)
		return actioned_crop_box

	def next_image(self):
		xml_image = self.xml_file.readline().strip()
		if xml_image!="":
			self.log_file.write("next image: %s\n"%(xml_image))
			self.xml_image = xml_image
			mirror = False
			if random.random() < 0.5 and not self.Test:
				mirror = True
			self.cur_image = Image.open(xml_image.replace("Annotations", "JPEGImages").replace("xml", "jpg"))
			if mirror:
				self.cur_image = self.cur_image.transpose(Image.FLIP_LEFT_RIGHT)
			xml_input = xml.dom.minidom.parse(xml_image)
			self.cur_size[0] = get_value_by_tag(xml_input, 'width')
			self.cur_size[1] = get_value_by_tag(xml_input, 'height')
			objs = xml_input.getElementsByTagName('object')
			self.cur_box = []
			self.other_box = []
			for obj in objs:
				tmp = [0,0,0,0]
				tmp[0] = get_value_by_tag(obj, 'xmin')
				tmp[1] = get_value_by_tag(obj, 'ymin')
				tmp[2] = get_value_by_tag(obj, 'xmax')
				tmp[3] = get_value_by_tag(obj, 'ymax')
				if mirror:
					tmp[0] = self.cur_size[0] - get_value_by_tag(obj, 'xmax')
					tmp[2] = self.cur_size[0] - get_value_by_tag(obj, 'xmin')
				if get_name_by_tag(obj, 'name') != self.class_name:
					self.other_box.append(tmp)
				else:
					self.cur_box.append(tmp)
			self.log_file.write("cur_size:%s, cur_box:%s\n"%(str(self.cur_size), str(self.cur_box)))
			return 0
		else:
			return 1

	def update_train_db(self, t, state, action):
		actioned_crop_box = self.cal_actioned_crop(action)
		new_overlap = get_overlapping_ratio(actioned_crop_box, self.cur_box)
		last_overlap = get_overlapping_ratio(self.crop_box, self.cur_box)
		self.overlap_ratio = last_overlap / 10

		self.states[self.tindex] = state
		if t == 0:
			self.masks[self.tindex] = 1
		else:
			self.masks[self.tindex] = 0
		self.actions[self.tindex][0] = action
		reward =  (new_overlap - last_overlap) * 2

		self.rewards[self.tindex][0] = reward
		self.tindex = (self.tindex + 1) % self.tmaxnum

		if (not self.tfull) and (self.tindex == 0):
			self.tfull = True
		self.log_file.write("IoU %f, after action %s, got reward %f\n"%(self.overlap_ratio, action, reward))

	def get_index(self):
		if not self.tfull:
			index = random.randint(0, self.tindex-2)		
		else:
			index = random.randint(0, self.tmaxnum-1)
		if index == self.tindex-1 or self.masks[(index+1) % self.tmaxnum] \
				or (self.tindex == 0 and index == self.tmaxnum-1):
			index = self.get_index()
		return index

	def draw_random_batch(self, batch_size):
		if (not self.tfull) and (self.tindex <= self.D_threshold):
			return -1
		if not self.tfull:
			batch_size = min(batch_size, self.tindex)	
		for i in range(batch_size):
			index = self.get_index()
			self.batch_states[i] = self.states[index]
			self.batch_next_states[i] = self.states[(index+1) % self.tmaxnum]
			self.batch_actions[i] = self.actions[index]
			self.batch_rewards[i] = self.rewards[index]

		return (self.batch_states, self.batch_actions, self.batch_rewards, self.batch_next_states)

	def save_crop_image(self, box, path="../log/test.jpg"):
		self.cur_image.crop(box).save(path)

	def save_train_db(self, path):
		pass
