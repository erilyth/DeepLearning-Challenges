import numpy as np
import lutorpy as lua

require("torch")


file_list = open('filelist.txt', 'r')

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "

key_idx = 1
dictionary = {}
for alph_idx in range(len(alphabet)):
	dictionary[alphabet[alph_idx]] = key_idx
	key_idx += 1

file_list = open('filelist.txt', 'r')

for file in file_list.readlines():
	file = file.split(' ')[0]
	save_file = file + '.npy'
	th_save_file = file + 'text.t7'
	file = file + '.txt'
	captions = open(file, 'r')
	# doclength is 31
	cur_cap = np.zeros([1,201,5], dtype=int)
	caption_idx = 0
	for caption in captions.readlines():
		caption = caption[:-2]
		cnt = 0
		for char_idx in range(len(caption)):
			cur_cap[0,cnt,caption_idx] = dictionary[caption[char_idx].lower()]
			cnt += 1
		caption_idx += 1
		if caption_idx > 4:
			break
	tensor_t = torch.fromNumpyArray(cur_cap)
	torch.save(th_save_file, tensor_t)
	np.save(save_file, cur_cap)
