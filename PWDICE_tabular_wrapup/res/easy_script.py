f = open("restotal.txt","r")
lines = f.readlines()
for line in lines:
    contents = line.split()
    print("det:", contents[0], contents[3])
for line in lines:
    contents = line.split()
    print("sto:", contents[0], contents[9])
