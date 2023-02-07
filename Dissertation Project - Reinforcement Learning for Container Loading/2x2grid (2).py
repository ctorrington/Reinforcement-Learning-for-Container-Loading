from cgitb import grey
import csv
import pandas
import numpy
import matplotlib.pyplot as plt
import copy
from StateTree import Node


# TODO randomize boxes later
# TODO allow for boxes to move within the state space - done
# TODO allow for the rotation of boxes. will need to go in states_transitions. some for loop rotataing once and checking compatibility - repeating until 4 rotations
# TODO later allow boxes to be enterd as [x, y] sized coordinates and be able to change them to a row. ie flatten() - done
# TODO figure out how to label the plots - done
# TODO check whether valueiteration2 is any different


# container dimensions
dimensions = 2
x_dimension = dimensions
y_dimension = dimensions

state_space_needed = False


if dimensions == 2:
    # boxes = numpy.array([[0,1,0,1], [1,0,0,0], [0,0,1,0]])
    boxes = [[1, 2], [1, 1], [1, 1]]
elif dimensions == 3:
    # boxes = numpy.array([[0,0,0,0,1,1,0,1,1], [1,0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0]])
    boxes = numpy.array([[2,2],[1,1],[1,1],[1,1],[1,1],[1,1]])


state = 0
action = 1

def g(a, z):
    for i in range(x_dimension * y_dimension):
        t = a.copy()
        t[i] = 1
        if t not in state_space:
            state_space.append(t)
        if z < ((x_dimension * y_dimension) - 1):
            g(t, z + 1)


# state setup
if dimensions == 2:
    state_space = [
        [0, 0, 0, 0],

        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],

        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 0],

        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0],

        [1, 1, 1, 1],
    ]

elif dimensions == 3:
    # need to create 3x3 state space
    if state_space_needed == True:
        state_space = [[0] * (x_dimension * y_dimension)]

        g(state_space[0], 0)
        df = pandas.DataFrame(state_space)
        df.to_csv("3x3_state_space.csv", index = False)



    state_space = []
    with open("3x3_state_space.csv", newline = "") as f:
        reader = csv.reader(f)
        for row in reader:
            for i in range(len(row)):
                row[i] = int(row[i])
            state_space.append(row)

    state_space.sort()


elif dimensions == 4:
    # TODO run the 4x4 grid for a few hours
    pass
elif dimensions == 5:
    # if time run 5x5 grid. will probably take days
    pass

# Value of each state
V = numpy.zeros(len(state_space))



# setup for possible_next_states dictionary
# TODO can probably change this enumerate to an in range - i never used
possible_next_states = {}
for c, i in enumerate(state_space):
        possible_next_states[c] = []


rewards = []
for count, i in enumerate(state_space):
    if sum(i) == (x_dimension * y_dimension):
        rewards.append(1)
    else:
        rewards.append(0)

# dict for boxes remaining per state setup
boxes_remaining = {}
boxes_remaining[0] = boxes


def box_translation(b):
    t = []
    # moving the box along the state space y axis, stopping short of the edge
    for j in range(0, y_dimension - b[1] + 1):
    # moving the box along the state space x axis, stopping short the edge
        for i in range(0, x_dimension - b[0] + 1):
            a = []
            
            # filling top rows once box has moved down
            if b[1] < y_dimension and j > 0:
                    for k in range(j):
                        a.append([0] * x_dimension)

            # placing the box on the current row and filling current row with empty spaces
            for z in range(b[1]):
                a.append([0] * i + [1] * b[0] + [0] * (x_dimension - i - b[0]))

            # filling bottom rows before box has moved down
            if b[1] < y_dimension and j + b[1] < y_dimension:
                    for k in range(y_dimension - b[1] - j):
                        a.append([0] * x_dimension)
            t.append(a)
    return t
                



# rewards for each state
# {0: [6, 4, 2], 1: [7, 5], 2: [11, 9], 3: [10, 8], 4: [13, 9], 5: [12], 6: [13, 11], 7: [12], 8: [14], 9: [15], 10: [14], 11: [15], 12: [], 13: [15], 14: [], 15: []}
# new and cooler rewards function
def set_rewards1(s):
    if rewards[s.data] < s.get_depth():
        rewards[s.data] = s.get_depth()
    for i in s.get_children(): 
        set_rewards1(i)


# def set_rewards(s, count):
#     # not in the first state
#     if s > 0:
#         # check whether more blocks are currently packed
#         if rewards[s] < count:
#             # increment the reward of the last state rather than overwriting
#             if s == len(state_space) - 1:
#                 rewards[s] = count + 1
#             else:
#                 rewards[s] = count
#     for i in possible_next_states[s]:
#         set_rewards(i, count + 1)




# Policy mapping actions to states
# pi is 2-dimensional array (i, j) where the probability of taking action j in state i
def set_policy():
    pi = numpy.zeros((len(state_space), len(state_space)))
    for c, i in enumerate(possible_next_states):
        for j in possible_next_states[i]:
            pi[c][j] = 1

    for c, i in enumerate(pi):
        with numpy.errstate(divide='ignore', invalid='ignore'):
            pi[c] = pi[c] / sum(pi[c])
            pi[numpy.isnan(pi)] = 0

    return pi






# function to print state values
def print_state_values():
    print("state values")
    for s, v in enumerate(V):
        print(f"State {s} has value of {v}")

# function to print the current policy
def print_policy():
    print("policy")
    for s, pi_s in enumerate(pi):
        print(f"".join(f"pi(A={a}|S={s}) = {p.round(2)}" + 4 * " " for a, p in enumerate(pi_s)))


def box_to_arrays(box):
    c = 0
    a = numpy.array([])
    tmp = numpy.array([])
    for i in box:
        a = numpy.append(a, i)
        c += 1
        if c == x_dimension:
            # np.stack((a, b))
            if tmp.size == 0:
                tmp = a
            else:
                tmp = numpy.stack((tmp, a))
            a = numpy.array([])
            c = 0
    return tmp

def rotate_clockwise(box):
    return numpy.rot90(box, 3)

def rotate_anticlockwise(box):
    return numpy.rot90(box)




# calc possible next states
def calc_possible_next_states(state_index, boxes_avail, current_node):
    #  # check whether boxes fit into state
    # for state_space_index, s in enumerate(state_space):
    # for every box remaining in each state
    for box_index_count, b in enumerate(boxes_avail): # change to 'boxes' to fix
        # box rotation
        tmp_x = b[1]
        tmp_y = b[0]

        is_rotation_necessary = False
        # check if rotation is necessary
        if tmp_x != tmp_y:
            is_rotation_necessary = True
            box_rotated = [tmp_x, tmp_y]

        
        # get every boxes possible placements within the packing space
        all_box_positions = box_translation(b)

        if is_rotation_necessary:
            all_box_positions.extend(box_translation(box_rotated))

        # print(f"rotation necessary: {is_rotation_necessary}")
        # print("all box positions")
        # print(all_box_positions)
        # input()

        for c, k in enumerate(all_box_positions):
            all_box_positions[c] = numpy.array(all_box_positions[c])
            all_box_positions[c] = all_box_positions[c].flatten()


        for a in all_box_positions:
            # place the first box into the state space
            test_state = state_space[state_index] + a

            is_overlap = False
            # check if overlap
            for z in test_state:
                if z > 1:
                    # there is overlap
                    is_overlap = True

            # there is no overlap - add the possible state
            if not is_overlap:
                # get the state index for the possible_next_states dictionary
                for count1, i in enumerate(state_space):
                    # finds the next states index
                    if numpy.array_equal(i, test_state):
                        # adding the next possible states as a list to the current state
                        possible_next_states[state_index].append(count1)
                        n = Node(count1)
                        current_node.addNode(n)                            
                        t = copy.deepcopy(boxes_avail)
                        t.pop(box_index_count)
                        n.set_boxes_avail(len(t))
                        calc_possible_next_states(count1, t, n)


# function to return the reward and probability of every possible state transition from the given state with selected action
# env transitions
def env_transitions(state, action):
    global rewards
    # # action parameter left in for legibility but has no value since environment will not impact it
    # tmp = []
    # # for all the possible next states from the given state
    # for s in possible_next_states[state]:
    #     # this is the environments impact on the agent. env does not impact agent so prob of next state is 1
    #     tmp.append([rewards[s], 1])
    # # return tmp


    # # TODO fix this. kinda working but not. need the p(s' = 0, r = x | s = 3, a = y) = somthin. check the ref link
    # tmp = []
    # for c, i in enumerate(state_space):
    #     tmp.append([rewards[c], pi[state][action]])

    # return tmp

    tmp = []
    # rewards for each state
    for i in possible_next_states:
        a = []
        a.append(rewards[i]) 
        # env state transitions. if its possible its 1. if its not possible its 0
        if i in possible_next_states[state] and i == action:
            a.append(1)
        else:
            a.append(0)
        tmp.append(a)

    return tmp







# Policy Evaluation
# policy eval sudo code from barto and sutton chapt 4
# NB!! DO NOT CHANGE c TO s. IT WILL LEAD TO HOURS OF DEBUGGING PAIN
def evaluate_policy(V, pi, gamma, theta):
    while True:
        delta = 0
        for c, s in enumerate(state_space):
            v = V[c] 
            bellman_update(V, pi, c, gamma) 
            delta = max(delta, abs(v - V[c])) 
        if delta < theta:
            break
    return V


def bellman_update(V, pi, s, gamma):
    v = 0
    # for every action
    for a, i in enumerate(state_space):
        transitions = env_transitions(s, a)
        # for every next state following action a
        for s_, (r, p) in enumerate(transitions):
            v += pi[s][a] * p * (r + gamma * V[s_])    
    V[s] = v


# policy iteration
# policy iteration sudo code from barto and sutton chapt 4
def improve_policy(V, pi, gamma):
    policy_stable = True
    for c, s in enumerate(state_space):
        old = pi[c].copy()
        q_greedify_policy(V, pi, c, gamma)
        if not numpy.array_equal(pi[c], old):
            policy_stable = False
    return pi, policy_stable
        

def policy_iteration(gamma, theta):
    V = numpy.zeros(len(state_space)) # dont know if this is necessary
    pi = set_policy()
    policy_stable = False
    while not policy_stable:
        V = evaluate_policy(V, pi, gamma, theta)
        pi, policy_stable = improve_policy(V, pi, gamma)
    return V, pi


def q_greedify_policy(V, pi, s, gamma):
    G = numpy.zeros(len(state_space))
    for a, i in enumerate(state_space):
        transitions = env_transitions(s, a)
        for s_, (r, p) in enumerate(transitions):
            G[a] += p * (r + gamma * V[s_])

    greedy_actions = numpy.argwhere(G == numpy.amax(G))
    for a, i in enumerate(state_space):
        if a in greedy_actions:
            pi[s, a] = 1 / len(greedy_actions)
        else:
            pi[s, a] = 0








# value iteration
# value iteration sudo code from barto and sutton chapt 4
def value_iteration(gamma, theta):
    V = numpy.zeros(len(state_space))
    while True:
        delta = 0
        for c, s in enumerate(state_space):
            v = V[c]
            bellman_optimality_update(V, c, gamma)
            delta = max(delta, abs(v - V[c]))
        if delta < theta:
            break
    pi = set_policy()
    for c, s in enumerate(state_space):
        q_greedify_policy(V, pi, c, gamma)
    return V, pi

def bellman_optimality_update(V, s, gamma):
    vmax = -float("inf")
    for a, i in enumerate(state_space):
        transitions = env_transitions(s, a)
        va = 0
        for s_, (r, p) in enumerate(transitions):
            va += p * (r + gamma * V[s_])
        vmax = max(va, vmax)
    V[s] = vmax





print("intial value function")
print_state_values()




### TESTING ###
# b2 = [0,0,1,0]
# print("b2")
# print(b2)
# b2 = box_to_arrays(b2)
# b2 = numpy.array(b2)
# print("box_to_arrays b2")
# print(b2)

# b2t = rotate_clockwise(b2)
# print("rotated b2")
# print(b2t)
# print("flattened rotated b2")
# print(b2t.flatten())

#################

state = 5


boxes_avail = copy.deepcopy(boxes)
root = Node(0)
print("next possible states from every state")
calc_possible_next_states(0, boxes_avail, root)
print(possible_next_states)

print("rewards")
# set_rewards(0, 0)
set_rewards1(root)
for c, i in enumerate(state_space):
    if sum(i) == x_dimension * y_dimension:
        rewards[c] += 1
print(rewards)




pi = set_policy()




# print("policy")
# for s, pi_s in enumerate(pi):
#     print(f"".join(f"pi(A={a}|S={s}) = {p.round(2)}" + 4 * " " for a, p in enumerate(pi_s)))


# print_policy() NB removed for testing ###############################

state = 0
action = 2

print(f"probability of choosing action {action} in state {state}: {pi[state][action]}")

print(f"env_transitions for given state {state} and given action {action}")
transitions = env_transitions(state, action)
print(transitions)

print("transitions for given action and state")
for s_, (r, p) in enumerate(transitions):
    print(f"p(S\'={s_}, R={r} | S={state}, A={action}) = {p}")


print("policy evaluation complete")
gamma = 1
theta = 0.1
V = evaluate_policy(V, pi, gamma, theta)
# print(V)
print_state_values()

# plot value function
# fig, ax = plt.subplots()
# ax.plot(V)
# plt.show()

# plot value function and policy after policy evaluation
fig, (ax1, ax2) = plt.subplots(2)
# plt.figure(figsize=(10, 10), dpi = 80)
fig.suptitle('value function and policy after policy evaluation')
ax1.plot(V)
ax1.set_title("value function")
ax1.set_xlabel("state")
ax1.set_ylabel("value")

ax2.imshow(pi)
ax2.set_title("policy")
ax2.set_xlabel("state")
ax2.set_ylabel("action/ next state")
# ax2.set_ylim(0, len(state_space) - 1)
pi_plot1 = ax2.imshow(pi)
bar = plt.colorbar(pi_plot1)


# ax3.plot(pi)
# ax3.set_title("policy")
# plt.show()
plt.draw()




print("policy iteration complete")
gamma = 1
theta = 0.1
V, pi = policy_iteration(gamma, theta)
print_state_values()

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('value function and policy after policy iteration')
ax1.plot(V)
ax1.set_title("value function")
ax2.imshow(pi)
ax2.set_title("policy")
ax2.set_xlabel("state")
ax2.set_ylabel("action/ next state")
pi_plot2 = ax2.imshow(pi)
bar = plt.colorbar(pi_plot2)
# plt.show()
plt.draw()

print_policy()





print("value iteration complete")
gamma = 1
theta = 0.1
V, pi = value_iteration(gamma, theta)
print_state_values()
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('value function and policy after value iteration')
ax1.plot(V)
ax1.set_title("value function")
ax2.imshow(pi)
ax2.set_title("policy")
ax2.set_xlabel("state")
ax2.set_ylabel("action/ next state")
pi_plot3 = ax2.imshow(pi)
bar = plt.colorbar(pi_plot3)
# plt.show()
plt.draw()
plt.show()