a = open("cfa2.dat","r")
b = open("homework.text","w")
a_line=a.readlines()[13:]


for i in a_line:
    print(i[0:36]+"\n")
