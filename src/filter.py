from data import get_overlapping_ratio, cal_box_area

class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
				'bus', 'car', 'cat', 'chair', 'cow', 
				'diningtable', 'dog', 'horse', 'motorbike', 'person',
				'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def any_overlap(box_list, box, thres):
	for elebox in box_list:
		if get_overlapping_ratio(elebox, box) > thres:
			return True
	return False

def filter(thres, scoreThres):
	for i in range(20):
		fin = open("../output/test/comp3_det_test_"+class_names[i]+".txt","r")
		fout = open("../output/test"+str(thres)+"/comp3_det_test_"+class_names[i]+".txt","w")
		line = fin.readline().strip()
		box = [0,0,0,0]
		while line != "":
			last_imgid = ""
			imgid = ""
			box_list = []
			loop = True
			have_one = False

			while line != "" and loop:
				print "#"+line + "#"
				splits = line.split(" ")
				imgid = splits[0]
				def score_map(score):
					return score
				score = score_map(float(splits[1]))
				box[0] = int(splits[2])
				box[1] = int(splits[3])
				box[2] = int(splits[4])
				box[3] = int(splits[5])
#				print box_list
#				print box
				if imgid == last_imgid or last_imgid == "":
					if any_overlap(box_list, box, thres) or score < scoreThres:
						pass
					else:
						tmp = [0,0,0,0]
						tmp[0] = box[0]
						tmp[1] = box[1]
						tmp[2] = box[2]
						tmp[3] = box[3]
						box_list.append(tmp)
						have_one = True
						fout.write(str(imgid)+ " " + str(score) + " " + 
									str(tmp[0]) + " " +str(tmp[1]) + " " +str(tmp[2]) + " " +str(tmp[3]) + " " +"\n")
					line = fin.readline().strip()
				else:
					loop = False
				last_imgid = imgid

if __name__ == "__main__":
	filter(3, 0.5)
