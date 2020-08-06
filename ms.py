#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import os, sys, time, datetime, json, random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import cv2
from p_fruit import contains_red_lichee
#get_ipython().run_line_magic('matplotlib', 'notebook')
cmap = colors.ListedColormap(['red', 'green','blue', 'cyan', 'gray', 'violet', 'white'])
boundaries = [0, 0.15, 0.25, 0.5, 0.6, 0.7, 0.8, 1]
norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

# In[2]:


maze = np.array([
[ 1., 0., 1., 1., 1., 1., 0., 1., 1., 1.],
[ 1., 1., 1., 1., 1., 0., 1., 1., 0., 1.],
[ 1., 1., 1., 1., 1., 0., 1., 1., 1., 1.],
[ 1., 0., 0., 0., 1., 1., 1., 1., 1., 0.],
[ 1., 1., 0., 1., 0., 1., 1., 0., 0., 1.],
[ 1., 1., 0., 1., 0., 1., 1., 1., 1., 1.],
[ 1., 1., 1., 1., 1., 0., 1., 1., 1., 1.],
[ 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
[ 1., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
[ 1., 1., 1., 1., 1., 1., 1., 0., 1., 1.]
])

# In[3]:


visited_mark = 0.8  # Cells visited by the rat will be painted by gray 0.8
rat_mark = 0.5      # The current rat cell will be painteg by gray 0.5
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

# Actions dictionary
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

num_actions = len(actions_dict)

# Exploration factor
epsilon = 0.1

intermediate_and_final=[[0,8],
                        [1,1],
                        [2,2],
                        [2,7],
                        [3,4],
                        [3,6],
                        [4,5],
                        [4,9],
                        [0,5],
                        [6,2],
                        [6,3],
                        [7,4],
                        [8,8],
                        [9,5],
                        [9,9]]


maze_img_map = np.zeros(maze.shape)
maze_img_map = maze_img_map.astype(np.str)
image_list = os.listdir("imgs/")
print(image_list)
cv2.namedWindow("licheeview",cv2.WINDOW_NORMAL)

cv2.resizeWindow("licheeview",800,600)

if (len(intermediate_and_final)-1)!=len(image_list):
    print("Please make sure number of image files in imgs folder equals to that of number of coordinates")
    sys.exit(0)

for coord,file in zip(intermediate_and_final[:-1],image_list):
    print(coord,file)
    maze_img_map[coord[0],coord[1]]=file
print(maze_img_map)

# In[4]:


# maze is a 2d Numpy array of floats between 0.0 to 1.0
# 1.0 corresponds to a free cell, and 0.0 an occupied cell
# rat = (row, col) initial rat position (defaults to (0,0))

class Qmaze(object):
    def __init__(self, maze, rat=(0,0)):
        self._maze = np.array(maze)
        nrows, ncols = self._maze.shape
        self.target = (nrows-1, ncols-1)   # target cell where the "cheese" is
        self.free_cells = [(r,c) for r in range(nrows) for c in range(ncols) if self._maze[r,c] == 1.0]
        self.free_cells.remove(self.target)
        if self._maze[self.target] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if not rat in self.free_cells:
            raise Exception("Invalid Rat Location: must sit on a free cell")
        self.reset(rat)

    def reset(self, rat):
        self.rat = rat
        self.maze = np.copy(self._maze)
        nrows, ncols = self.maze.shape
        row, col = rat
        self.maze[row, col] = rat_mark
        self.state = (row, col, 'start')
        self.min_reward = -0.5 * self.maze.size
        self.total_reward = 0
        self.visited = set()

    def update_state(self, action):
        nrows, ncols = self.maze.shape
        nrow, ncol, nmode = rat_row, rat_col, mode = self.state

        if self.maze[rat_row, rat_col] > 0.0:
            self.visited.add((rat_row, rat_col))  # mark visited cell

        valid_actions = self.valid_actions()
                
        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == LEFT:
                ncol -= 1
            elif action == UP:
                nrow -= 1
            if action == RIGHT:
                ncol += 1
            elif action == DOWN:
                nrow += 1
        else:                  # invalid action, no change in rat position
            mode = 'invalid'

        # new state
        self.state = (nrow, ncol, nmode)

    def get_reward(self):
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows-1 and rat_col == ncols-1:
            return 1.0
        if mode == 'blocked':
            return self.min_reward - 1
        if (rat_row, rat_col) in self.visited:
            return -0.25
        if mode == 'invalid':
            return -0.75
        if mode == 'valid':
            return -0.04

    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return envstate, reward, status

    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    def draw_env(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r,c] > 0.0:
                    canvas[r,c] = 1.0
        # draw the rat
        row, col, valid = self.state
        canvas[row, col] = rat_mark
        return canvas

    def game_status(self):
        if self.total_reward < self.min_reward:
            return 'lose'
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows-1 and rat_col == ncols-1:
            return 'win'

        return 'not_over'

    def valid_actions(self, cell=None):
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        actions = [0, 1, 2, 3]
        nrows, ncols = self.maze.shape
        if row == 0:
            actions.remove(1)
        elif row == nrows-1:
            actions.remove(3)

        if col == 0:
            actions.remove(0)
        elif col == ncols-1:
            actions.remove(2)

        if row>0 and self.maze[row-1,col] == 0.0:
            actions.remove(1)
        if row<nrows-1 and self.maze[row+1,col] == 0.0:
            actions.remove(3)

        if col>0 and self.maze[row,col-1] == 0.0:
            actions.remove(0)
        if col<ncols-1 and self.maze[row,col+1] == 0.0:
            actions.remove(2)

        return actions


# In[5]:


def show(qmaze,tree_points):
    plt.grid('on')
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    for row,col in qmaze.visited:
        canvas[row,col] = 0.6
    rat_row, rat_col, _ = qmaze.state
    canvas[rat_row, rat_col] = 0.3   # rat cell
    for tp in tree_points[:-1]:
        nr,nc=tp
        canvas[nr, nc] = 0.2
        if rat_row==nr and rat_col==nc:
            canvas[nr, nc] = 0.75

    canvas[nrows-1, ncols-1] = 0.9 # cheese cell
    if rat_row==nrows-1 and rat_col==ncols-1:
        canvas[rat_row, rat_col] = 0.3
    img = plt.imshow(canvas, interpolation='none', cmap=cmap, norm=norm)
    return img


# In[6]:
from collections import defaultdict

def bfs(start_point,dest_point):
    cur_point=[start_point[0],start_point[1]]
    y_pt=[-1,0,0,1]
    x_pt=[0,-1,1,0]
    queue=[start_point]
    i=0
    visited=[start_point]
    qpts=np.zeros(maze.shape+(2,),dtype=np.int8)
    while cur_point!=dest_point:
        for pty,ptx in zip(y_pt,x_pt):
            #print(i)
            cy,cx=queue[i][0]+pty,queue[i][1]+ptx
            if cy>=0 and cx>=0 and cx<10 and cy<10:
                pt_temp=[cy,cx]
                if maze[cy,cx]==1:
                    if [cy,cx] not in visited:
                        queue.append([cy,cx])
                        visited.append([cy,cx])
                        qpts[cy,cx]=queue[i]
                        #print(queue[i])
                        if [cy,cx]==dest_point:
                            cur_point=[cy,cx]
                            break
        i+=1
    cp=[dest_point[0],dest_point[1]]
    ret_lst=[[dest_point[0],dest_point[1]]]
    #print(ret_lst)
    while cp!=[start_point[0],start_point[1]]:
        cp=list(qpts[cp[0],cp[1]])
        ret_lst.append(cp)
    #print(ret_lst)
    return [ret_lst[i] for i in range(len(ret_lst)-1,-1,-1)]




import time
def move_bot(qmaze,intermediate_stops):
    curpos=[0,0]
    start_point=[curpos[0],curpos[1]]
    fig=plt.figure()
    ax=fig.add_subplot(111)
    for i_stop in intermediate_stops:
        #print(start_point)
        destination=[i_stop[0],i_stop[1]]
        path_plan=bfs(start_point,destination)
        #print(path_plan)
        action=None
        
        for item in path_plan:
            ax.cla()
            action=None
            s_y,s_x=item[0]-curpos[0],item[1]-curpos[1]
            if s_y<0:
                action=UP
            elif s_y>0:
                action=DOWN
            elif s_x<0:
                action=LEFT
            elif s_x>0:
                action=RIGHT
            if action is not None:
                _,_,game_over = qmaze.act(action)
            curpos=[item[0],item[1]]
            #print(curpos)
            show(qmaze,intermediate_stops)
            fig.canvas.draw()
            img=os.path.join("imgs",maze_img_map[curpos[0],curpos[1]])
            print(curpos,img)
            if os.path.exists(img):
                contains_red_lichee(img,"licheeview")
                
            else:
                cv2.imshow("licheeview",np.zeros((800,600)))
            plt.pause(1)
            maze_img_map[curpos[0],curpos[1]]='0'
        start_point=[destination[0],destination[1]]
    plt.ioff()
    plt.show()


# In[78]:

import math
import copy
qmaze = Qmaze(maze)


def calculate_checkpoint_serial(maze,intermediate_points):
    #get the closest point to origin 
    dist_min=100000000000
    serialized_path=[]
    fp=None
    initial_point=[0,0]
    ips=copy.copy(intermediate_points)
    #print(ips)
    while(len(ips)>1):
        dist_min=100000000000
        for i_pt in ips[:-1]:
            
            dist=len(bfs(initial_point,i_pt))
            
            if dist<dist_min:
                dist_min=dist  
                fp=i_pt
            #print(dist,dist_min,i_pt)

        serialized_path.append(fp)
        #print("point",fp)
        ips.remove(fp)
        #print("list:",ips)
        initial_point=fp
    serialized_path.append(intermediate_points[-1])
    return serialized_path,0

import itertools
def calculate_ckpt(maze,intermediate_points):
    total_dist=1000000000000000
    seq=None
    for point in itertools.permutations([[0,0]]+intermediate_points,len(intermediate_points)+1):
        if point[0]==[0,0] and point[len(point)-1]==intermediate_points[-1]:
            dist=0
            for i in range(0,len(point)-1):
                start_point=point[i]
                end_point=point[i+1]
                dist+=len(bfs(start_point,end_point))
            #print(point,dist)
            if dist<=total_dist:
                total_dist=dist
                seq=point
    return seq[1:],total_dist
#calculate_ckpt(maze,[[6,9],[3,5],[1,2],
#                                  [2,4],
#                                  [5,0],
#                                  [7,4],
#                                  [9,9]])


def calculate_ckpt_cluster(maze,intermediate_points):
    sl=sorted(intermediate_points,key=lambda l:l[0], reverse=False)
    sl_up=[[0,0]]+sl[0:len(sl)//2]
    sl_down=sl[(len(sl)//2)+1:]
    seq=None
    
    total_dist=100000000000
    for point in itertools.permutations(sl_up,len(sl_up)):
        dist=0
        if point[0]==[0,0]:
            for i in range(0,len(point)-1):
                start_point=point[i]
                end_point=point[i+1]
                dist+=len(bfs(start_point,end_point))
            if dist<=total_dist:
                total_dist=dist
                seq=point
    total_dist=100000000000
    seq_2=None
    for point in itertools.permutations(sl_down,len(sl_down)):
        dist=0
        if point[-1]==[9,9]:
            for i in range(0,len(point)-1):
                start_point=point[i]
                end_point=point[i+1]
                dist+=len(bfs(start_point,end_point))
            if dist<=total_dist:
                total_dist=dist
                seq_2=point
    return seq+seq_2,0



#sp=calculate_checkpoint_serial(maze,intermediate_and_final)
#print(sp)
#move_bot(qmaze,sp)
#img=show(qmaze,intermediate_and_final)
#plt.show()



sp=calculate_ckpt_cluster(maze,intermediate_and_final)
print(sp)
move_bot(qmaze,sp[0])



# In[ ]:




