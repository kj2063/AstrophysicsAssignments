import matplotlib.pyplot as plt
import numpy as np
import camb
from camb import model, initialpower
from scipy import optimize

a = [1,2,3]

# def perm(n,k):
#     if k == n:
#         print(a)
#     else:
#         for i in range(k,n):
#             a[k],a[i] = a[i],a[k]
#             perm(n,k+1)
#             # a[k],a[i] = a[i],a[k]
############
# import itertools

# result = itertools.permutations(a)

# print(list(result))
############

# print(perm(3,0))

# print(list(range(0,4)))
##################
# arr = [2,3,4,5]
# bit = [0]*len(arr)

# for i in range(2):
#     bit[0] = i
#     for j in range(2):
#         bit[1] = j
#         for k in range(2):
#             bit[2] = k
#             for l in range(2):
#                 bit[3] = l
#                 print([arr[x] for x in range(len(bit)) if bit[x]])
#                 # print(bit)
                
############
arr = [2,3,4,5]
n = len(arr)

print(list(range(1<<4)))
# for i in range(1<<n):
    # print(i)