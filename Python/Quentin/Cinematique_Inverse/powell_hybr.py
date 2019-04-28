# -*- coding: utf-8 -*-

# packages
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

def f(theta):
    return [arm_len[0]*np.cos(theta[0]) + arm_len[1]*np.cos(sum(theta)), 
            arm_len[0]*np.sin(theta[0]) + arm_len[1]*np.sin(sum(theta))] 

def jac(theta):
    return np.array([
        [-arm_len[0]*np.sin(theta[0]) - arm_len[1]*np.sin(sum(theta)), -arm_len[1]*np.sin(sum(theta))], 
        [arm_len[0]*np.cos(theta[0]) + arm_len[1]*np.cos(sum(theta)), arm_len[1]*np.cos(sum(theta))]
        ])
    

def func2(theta):
    f = [arm_len[0]*np.cos(theta[0]) + arm_len[1]*np.cos(sum(theta)) - pos[0], 
         7 + arm_len[0]*np.sin(theta[0]) + arm_len[1]*np.sin(sum(theta)) - pos[1]] 
    
    df = np.array([
        [-arm_len[0]*np.sin(theta[0]) - arm_len[1]*np.sin(sum(theta)), -arm_len[1]*np.sin(sum(theta))], 
        [arm_len[0]*np.cos(theta[0]) + arm_len[1]*np.cos(sum(theta)), arm_len[1]*np.cos(sum(theta))]
        ])

    return f,df

def draw(theta, color='-or'):
    x = [0,
         arm_len[0]*np.cos(theta[0]),
         arm_len[0]*np.cos(theta[0]) + arm_len[1]*np.cos(sum(theta))]
    y = [7,
         7 + arm_len[0]*np.sin(theta[0]),
         7 + arm_len[0]*np.sin(theta[0]) + arm_len[1]*np.sin(sum(theta))]
    print("Norm de l1 : {}, norme de l1+l2 : {}".format(np.linalg.norm([x[1],y[1]-7]), np.linalg.norm([x[2],y[2]])))

    plt.plot(x,y, color)
#theta = root(fun, [0, 0], jac=jac, method='hybr')
#plt.ion()

#plt.figure(1)
#plt.subplot(111)

def findTheta(depth=0):
    if depth != 0:
        global arm_len
        arm_len = [14.5, 18.5]
        global pos
        pos = np.array([depth,11])
        sol = root(func2, [0.5, 0.5], jac=True, method='hybr')
        theta = np.round(sol.x, 3)
        if theta[0] < 0:
            r = np.linalg.norm(pos - np.array([0, 7]))
            alpha = np.arcsin((pos[1]-7)/r)
            theta[0] = - theta[0] + 2*alpha
            theta[1] = - theta[1] #- 2*alpha
        return theta
    else:
        print("Retour à la position d'équilibre")
        return [0,0] # à définir
    

"""
global arm_len
arm_len = [14.5, 18.5]
global pos
pos = np.array([6,11])
for t in np.arange(6, 32, 2):
    pos[0] = t
    pos[1] = 11
    sol = root(func2, [0.5, 0.5], jac=True, method='hybr')
    #sol.success
    theta = np.round(sol.x, 3)
    print("pos = [{}, {}]".format(pos[0],pos[1]))
    plt.plot(pos[0],pos[1], marker='o', markersize=15, color='g')
    draw(theta, color='-ob')
    if theta[0] < 0:
        r = np.linalg.norm(pos - np.array([0, 7]))
        alpha = np.arcsin((pos[1]-7)/r)
        theta[0] = - theta[0] + 2*alpha
        theta[1] = - theta[1] #- 2*alpha
        draw(theta)
plt.plot([-3, -3, 3, 3], [0, 6.5, 6.5, 0])
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, axis='both')
plt.show()
    
#plt.ioff()
"""




