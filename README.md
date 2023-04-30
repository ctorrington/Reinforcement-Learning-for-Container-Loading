# Reinforcement-Learning-for-Container-Loading

This is my dissertation project that I completed at the University of Greenwich in 2022.

## Abstract

Recent studies on solving the container loading problem utilise heuristics as solution methods. Heuristics offer computationally less expensive solutions, however, they are not optimal solutions. These heuristical solution methods follow hand written rules that could be inhibiting their performance. In this project, we present a reinforcment learning algorithm that is capable of obtaining multipe optimal solutions for the container loading problem. The problem is modelled as a Markov Decision Process with an environment which is repesented by a tree of obtainable next states and boxes still to be packed. A policy is used for selecting which boxes to pack and their placement within the container. This policy is evaluated and iterated upon unti an optimal state value function has been assigned to every state. This method allwos for the container loading problem to be represented as a finite, episodic task for a reinforcement learning agent to learn an optimal solution method.

## Introduction

The container loading problem (CLP) defines loading a set of boxes without overlap & within the confines of the container, into a two dimensional container such that the volume of the packed elements are maximised. Container loading problems, along with knapsack problems, bin packing problems, cutting-stock problems, strip packing problems & pallet loading problems are subsets of the cutting & packing problem. Cutting & packing problems have multiple real world applications within the cargo transportation & medical & warehouse packing industries as well as others. Improvements iin the single container loading problem are potentially transferable to other problems as the single container problem is the basis for many of the other problems, with solution methods of multiple containers expanding the problem into multiple single container problems. While hueristics are a common approach to the problem, they are unable to solve the problem to optimality. This paper presents a reinforcement learning agent capable of providing an optimal solution method.

## Project Idea & Objectives

The goal of the project was to investigate whether improvements to the current state-of-the-art solution methods for the CLP are possible. The current solution methods use handwritten algorithms for heuristic solutions. An optimal solution will be achieved once policy iteration converges the policy to the optimal policy, $\pi_*$, & the state value function to the optimal state value function, $v_*$.

<ol>
	<li>Represent problem as a Markov Decision Process for a finite episodic task.</li>
	<li>Generate possible next states tree.</li>
	<li>Environmnet transitions function from possible next states tree.</li>
	<li>Reward function for maximising boxes places & space used.</li>
	<li>Policy evaluation for the Markov Decision Process.</li>
	<li>Policy iteration to determine $\pi_*$ & $v_*$.</li>
</ol>

## Methodology
### Possible Next States
<p>
The possible next states $P A(s)$ for state $s$ are defined by the recursive function below. The function creates a tree structure to store the path taken after every box is placed. This is necessary to prevent a single box from being packed multiple times. Every node has the current state & boxes remaining to be packed.
</p>

$$let G be a tree so that G = G(V, E), where V = G(P A(G[s], E))$$

## Reward Function
<p>
Two reward functions are used: 1) rewards the agent for packing more boxes, defined below, & 2) for reaching the fully packed state. The reward function determines the behaviour of the agent & can be changed to suite different tasks. Reward function 1) gives a higher reward to states where more boxes can be packed in the future. This increases the space used but also prevents very large boxes from being packed & preventing multiple other boxes from being packed.
</p>

$$reward[s] = max(depth(G[s]), ∀s \in P A (G[s])), ∀ V (G[s])$$

## Environment Transitions
<p>
The agent interacts with the environment after every action. However, because the agent is in a controlled environment with no uncertainty, the environment transitions are derived from the possible next states tree. This is done to insert the possible next states into the Bellman equation, without needing additional passes over the state space. The environment transitions function is defined below.
</p>

$$
p = \begin{cases} 0 & a=1\\ 1 & b=2 \end{cases}
$$

## Policy Evaluation
<p>
The policy is evaluated to determine the state value function for the policy, $v_\pi$. The initial policy is an equiprobable policy & so the agent will choose between states of equal value randomly. $v_\pi$ is calculated iteratively with the Bellman eqauation, defined below, as an update rule. The state value function is used to determine better policies in policy iteration.
</p>

$$v_\pi(s) = \sum_{a}{\pi(a | s)}\sum_{s', r}{p(s', r | s, a)}[r + \gamma v_{\pi}(s')]$$

## Policy Iteration
<p>
Once $v_{\pi}$ has been determined the goal of policy iteration is to improve the policy to an optimal policy, $\pi_{*}$, & hence, calculate the optimal state value function, $v_{*}$. This is done by the policy choosing states with higher values, evaluating the policy with the values against the old policy, until there is no longer any improvements.
</p>

## Results
<p>
The solution, when run on a container of dimensions (2, 2) with boxes to pack of dimensions (1, 2), (1, 1), (1, 1), provides the following results:
</p>

![Value Function & Policy after Policy Evaluation](https://github.com/ctorrington/Reinforcement-Learning-for-Container-Loading/blob/main/Images/Value%20Function%20&%20Policy%20after%20Policy%20Evaluation.png?raw=true)
<p>
<b>Figure 1:</b> Value function (left) & policy $\pi$ (right) after policy $\pi$ evaluation.
</p>

![Value Function & Policy after Policy Iteration](https://github.com/ctorrington/Reinforcement-Learning-for-Container-Loading/blob/main/Images/Value%20Function%20&%20Policy%20after%20Policy%20Iteration.png?raw=true)

<p>
<b>Figure 2:</b> Value function (left) & policy $\pi$ (right) after policy $\pi$ iteration, resulting in $v_{*}$ (left) & $\pi_{*}$ (right).
</p>

<p>
Fig. 1 shows the equiprobable policy $\pi$ for all possible next states frorm $s$, the states with a probablity of zero are not possible next states. The policy $\pi$ is capable of optimality, however, it is not guaranteed. The policy is improved to $\pi_{*}$ after policy iteration in Fig. 2. This policy will guarantee optimality as the policy only selects actions that have the highest expected return.
</p>

## Evaluation
<p>
The solution method presented is capable of solving all paths to optimality, depending on the reward function. The reward function is configurable to reflect the requirements of the solution method. Multiple paths to optimality are presented, allowing for variation within the packing procedure. Only one iteration of policy improvement was necessary for the value function to converge to $v_{*}$. The state value function is seen updating from Fig. 1 to Fig. 2, these values are expected as the policy in Fig. 2 now selects actions that pack the maximum amount of boxes while reaching the final, fully packed state.
</p>

## Conclusions
 - Reinforcement learning is capable of returning an optimal solution method for the container loading problem.
 - The solution method works for all container sizes & box configurations.
 - When used for large scale problems the solution method requires large amounts of time for $v$ to reach convergence.
 - The solution method does not scale to real world problem sizes & therefore is not comparable to current heuristic approaches.
 
 ## Limitations & Future Work
 <p>
 The state space of the problem grows exponentially to the dimensions of the container due to the agent using a tabular solution method. For the state space to be determined, multiple passes over the dimensions of the container need to be completed & all states recorded. While this works for smaller problems, the time requirement for real world problems of large scale is not feasible. Future work could produce a heuristic approach where a reinforcement learning agent uses approximation functions rather than a tabular solution method to remve the need for very large state spaces & potentially match the current hand-written solution methods in computation time.
 </p>
 
