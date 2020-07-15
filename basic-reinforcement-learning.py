# basic-q-learning

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import time

edges = [(0, 4), (4, 3), (4, 5), (1, 5), (1, 3), (2, 3), (5, 5)] 
reward = []
x_data = []
y_data = []
goal = 5
episodes = 200
epoch = 0
gamma = 0.40
MATRIX_SIZE = 6 # Should be number of nodes + 1

#
# Plotting reward value increases
#
fig = plt.figure() 
ax1 = fig.add_subplot(1,1,1)

def animate(i):
	if(i < episodes):
		x_data.append(i)
		y_data.append(reward[i])
		ax1.clear()
		ax1.plot(x_data, y_data)

		plt.xlabel('Epoch')
		plt.ylabel('Reward')
		plt.title('Live graph Q-Matrix reward')	

# Create the R matrix of -1s
R = np.matrix(np.ones(shape =(MATRIX_SIZE, MATRIX_SIZE))) 
R *= -1

#
# Returns the next possible state, picked randomly, given the current
#
def next_action(state):
	while True:
		selected_state = np.random.randint(0, MATRIX_SIZE)
		if(R[state, selected_state] != -1):
			return selected_state
	return -1
		
#
# Returns the max Q value for the possible actions for the given state
#
def max_q_next_state_all_actions(state, actions_list):
	max_val = 0
	for action in actions_list:
		max_val = max(Q[state, action], max_val)
	return max_val

#
# Returns a list of all possible actions that can be taken for the given state
# 
def all_actions(state):
	result = []
	for i in range (0, MATRIX_SIZE):
		if(R[state, i] != -1):
			result.append(i)
	return result

#
# Return the path of maximum reward from "state" to goal
#
def maxpath(state):
	result = [state]
	iteration_limit = 10
	iterations = 0
	while(state != goal):
		maxval = -1
		maxindex = -1
		for i in range(0, MATRIX_SIZE):
			if(Q[state, i] > maxval):
				maxval = Q[state, i]
				maxindex = i
		if(maxindex > -1):
			state = maxindex
			result.append(state)
		iterations += 1
		if(iterations > iteration_limit):
			return result
	print (result)
	return result

# Configure R with the possible actions
for point in edges:
	if (point[1] == goal):
		R[point[0], point[1]] = 100
	else:
		R[point[0], point[1]] = 0
	if (point[0] == goal):
		R[point[1], point[0]] = 100
	else:
		R[point[1], point[0]] = 0
print (R)

#
# Create the Q matrix of 0s
#
Q = np.matrix(np.zeros(shape =(MATRIX_SIZE, MATRIX_SIZE))) 

print (Q)

#
# Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
# 

for i in range(1, episodes):
	current_state = np.random.randint(0, MATRIX_SIZE)
	print ("Episode " + str(i))
	
	while(True):
		future_action = next_action(current_state)
		print (" --- Current state = " + str(current_state) + ". Future action is " + str(future_action))
		potential_actions = all_actions(future_action)
		#
		# Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
		#
		Q[current_state, future_action] = R[current_state, future_action] + gamma * max_q_next_state_all_actions(future_action, potential_actions)	
		current_state = future_action

		if(current_state == goal):
			break
			
	reward.append((np.sum(Q/np.max(Q)*100)))
	epoch = epoch + 1
	#os.system('cls')
	#print ("Episodes completed = " + str(i))
	print (Q, end=' ')
	
print (len(reward))

# 
# Testing the validity of the Q matrix by checking reach to goal from 2
#
maxpath(2)

Q = Q*100/np.max(Q)
print (Q)

# Show the animation of the reward function
ani = animation.FuncAnimation(fig, animate, interval=330, repeat=False) 
plt.show()