index = 10

fileString = "output_spd_%d.txt"%(index)
file = open(fileString, "r")

output1String = "example%d.txt"%(index)
output1 = open(output1String, "w")

output2String = "example%dmario.txt"%(index)
output2 = open(output2String, "w")

lines = file.readlines()

line1 = lines[0].strip("\n")
line2 = lines[1].strip("\n")

output1.write(line1)
output2.write(line2)