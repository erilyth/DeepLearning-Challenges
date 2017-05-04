import numpy as np
import lutorpy as lua

require("torch")


file_list = open('filelist.txt', 'r')

dictionary = {}

cnt = 0

dictionary['.'] = cnt
cnt += 1
dictionary[','] = cnt
cnt += 1

for file in file_list.readlines():
	file = file[:-1] + '.txt'
	captions = open(file, 'r')
	for caption in captions.readlines():
		caption = caption[:-2]
		for word in caption.split(' '):
			if word not in dictionary:
				dictionary[word] = cnt
				cnt += 1

#print dictionary

file_list.close()
file_list = open('filelist.txt', 'r')

for file in file_list.readlines():
	save_file = file[:-1] + '.npy'
	th_save_file = file[:-1] + '.t7'
	file = file[:-1] + '.txt'
	captions = open(file, 'r')
	# doclength is 31
	cur_cap = np.zeros([1,51,5], dtype=int)
	caption_idx = 0
	for caption in captions.readlines():
		caption = caption[:-2]
		cnt = 0
		for word in caption.split(' '):
			cur_cap[0,cnt,caption_idx] = dictionary[word]
			cnt += 1
		caption_idx += 1
		if caption_idx > 4:
			break
	tensor_t = torch.fromNumpyArray(cur_cap)
	torch.save(th_save_file, tensor_t)
	np.save(save_file, cur_cap)
