import numpy as np


#Q1
a = open("cfa2.dat","r")
b = open("homework.text","w")
a_line=a.readlines()[13:]


for i in a_line:
    if i[31:36] == '     ':
        pass
    
    else:
        name = i[0:11]
        
        RA = float(i[11:13])*15 + float(i[13:15])*15/60 + float(i[15:19])*15/3600 
        
        if i[19] == "-":
            DEC = -(float(i[20:22]) + float(i[22:24])/60 + float(i[24:26])/3600)
        else:
            DEC = float(i[20:22]) + float(i[22:24])/60 + float(i[24:26])/3600
    
        if float(i[31:36]) >= 0:
            DIS = float(i[31:36])/70
        else:
            DIS = 0

    
    
    b.write("{} {:6.2f} {:6.2f} {:6.2f}\n".format(name,RA,DEC,DIS))

a.close()
b.close()

#Q2

b = open("homework.text","r")
    
b_line = b.readlines()

q_list = []

for i in b_line:
    q_list.append(float(i[26:34]))
    
Max_dline = q_list.index(max(q_list))

print("Farthest galaxy's Name/RA/DEC/Distance =", b_line[Max_dline])

#Q3

result_N = []

for i in b_line:
    
    R = float(i[26:34])*np.cos(np.pi*float(i[20:26])/180)
    Z = float(i[26:34])*np.sin(np.pi*float(i[20:26])/180)
    
    if R > 150 or Z >50:
        pass 
    else:
       result_N.append(i)

print("The number of galaxy at R<150 and Z<50 =",len(result_N))







