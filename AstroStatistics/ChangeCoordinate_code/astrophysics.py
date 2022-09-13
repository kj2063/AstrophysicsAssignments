#transform cartesian to spherical

import numpy as np

a = open("gal_coordinates_final.txt", "r")
b = open("transform_shperical.txt", "w")
a_line = a.readlines()

x=[]
y=[]
z=[]

for i in a_line:
    x.append(i.split(" ")[0])
    y.append(i.split(" ")[1])
    z.append(i.split(" ")[2])

b.write("{} {} {}\n".format("z","ra","dec"))
    
for i in range(69409):
    zr = np.sqrt(float(x[i])**2 + float(y[i])**2 + float(z[i])**2)*0.7*70/(3e5)
    ra = np.arccos(float(x[i]) / np.sqrt(float(x[i])**2 + float(y[i])**2))*180/np.pi
    dec = np.arccos(np.sqrt(float(x[i])**2 + float(y[i])**2) / np.sqrt(float(x[i])**2 + float(y[i])**2 + float(z[i])**2))*180/np.pi
        
    b.write("{:.7f} {:.7f} {:.7f}\n".format(zr,ra,dec))

a.close()
b.close()

#make random catalog
    
c = open("random_catalog.txt", "w")

c.write("{} {} {}\n".format("z","ra","dec"))

for i in range(69409*20):
    zr_r = np.random.uniform(0,0.2781)
    ra_r = np.random.uniform(0,90)
    dec_r = np.random.uniform(0,90)

    c.write("{:.7f} {:.7f} {:.7f}\n".format(zr_r,ra_r,dec_r))
    
c.close()