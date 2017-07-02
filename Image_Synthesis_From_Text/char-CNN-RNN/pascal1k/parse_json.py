import json

filelist = open('filelist.txt', 'w')

classlist = open('classlist.txt', 'r')

class_dict = {}
class_idx = 1
for classx in classlist.readlines():
	class_dict[classx[:-1]] = class_idx
	class_idx += 1
classlist.close()

print class_dict

with open('data.json') as data_file:
	data = json.load(data_file)
	for key in data:
		f1 = open(key + '.txt', 'w')
		for val in data[key]:
			f1.write(val+'\n')
		f1.close()
		classxf = 1
		classlist = open('classlist.txt', 'r')
		for classx in classlist.readlines():
			if classx[:-1] in key:
				break 
			classxf += 1
		classlist.close()
		filelist.write(key+' '+str(classxf)+'\n')

filelist.close()