import numpy as np
import matplotlib.pyplot as plt

#####Q1

#make "year" list
year = []

first_year = 1980

while first_year < 2021:
    
    year.append(first_year)
    
    first_year += 1

#make "GDP" list
GDP = []

a = open("Q1data.txt", "r")

line = a.readlines()[0]

for i in range(41):
    GDP.append(float(line.split("\t")[i]))
    
a.close()    
    
#make "rank" list
rank = []

a = open("Q1data.txt", "r")

line2 = a.readlines()[2]

for i in range(41):
    rank.append(float(line2.split(" ")[i]))
    
a.close()

fig, ax = plt.subplots(1,2)

ax[0].plot(year,GDP)
ax[0].set_xlabel('year')
ax[0].set_ylabel('unit :(ten thousand Won)')

ax[1].plot(year,rank)
ax[1].set_xlabel('year')
ax[1].set_ylabel('rank')
plt.ylim(30, 1)

plt.show()


#####Q2

b = open("homework.text","r")
    
galaxies_data = b.readlines()


#if you want to fix selected_galaxy, use [np.random.Randstate(seed).randint(0,18203)]
selected_galaxy = galaxies_data[np.random.randint(0,len(galaxies_data))]

print("2.selected galaxy name : {}".format(selected_galaxy[0:11]))

ra_c = np.pi*float(selected_galaxy[12:17])/180
dec_c = np.pi*float(selected_galaxy[19:24])/180
r_c = float(selected_galaxy[26:32])

num_neighbor = 0

fig = plt.figure(figsize=(9,12))
ax = fig.add_subplot(1,1,1, projection= '3d')

for i in galaxies_data:
    ra = np.pi*float(i[12:17])/180
    dec = np.pi*float(i[19:24])/180
    r = float(i[26:32])
    
    xx = np.sqrt((r_c*np.cos(dec_c)*np.cos(ra_c) - r*np.cos(dec)*np.cos(ra) )**2 + (r_c*np.cos(dec_c)*np.sin(ra_c) - r*np.cos(dec)*np.sin(ra))**2)
    yy = r*np.sin(dec) - r_c*np.sin(dec_c)
    
    distance = np.sqrt(xx**2 + yy**2)
    
    if distance > 10 :
        pass
    else:
        num_neighbor += 1
        
        x = r*np.cos(dec)*np.cos(ra)
        y = r*np.cos(dec)*np.sin(ra)
        z = r*np.sin(dec)
   
        
        ax.scatter(x, y, z, s=5, c="b")
        
x_c = r_c*np.cos(dec_c)*np.cos(ra_c)
y_c = r_c*np.cos(dec_c)*np.sin(ra_c)
z_c = r_c*np.sin(dec_c)

ax.scatter(x_c, y_c, z_c, s=5, c="r")

plt.show()

b.close()

print("  the number of neighbor galaxies(r < 10Mpc) : {}\n".format(num_neighbor-1))


#####Q3

N = 1000000
NN = 100000000

def pi_result(n):
    xxx = np.random.uniform(-1,1,n)
    yyy = np.random.uniform(-1,1,n)

    rr = np.sqrt(xxx**2 + yyy**2)
    
    number_inside = len(rr[rr<1])
    
    result = 4*number_inside/n
    return result

result_N = pi_result(N)
result_NN = pi_result(NN)
    
print("3.calculated pi (N = 10**6) = {}".format(result_N))
print("discrepancy from real value = {}\n".format(np.pi - result_N))

print("  calculated pi (N = 10**8) = {}".format(result_NN))
print("discrepancy from real value = {}".format(np.pi - result_NN))