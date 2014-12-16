import sys

train_file = sys.argv[1]
tf_file = sys.argv[2]
tl_file = sys.argv[3]

fr = open(train_file)

ff = open(tf_file,"wb")
fl = open(tl_file,"wb")
for line in fr.readlines():
	idx = line.rfind(" ")
	f = line[0:idx]
	l = line[idx:len(line)]
    data.append(f)
    label.append(l)
ff.writelines(data)
fl.writelines(label)
ff.close()
fl.close()
fr.close()
