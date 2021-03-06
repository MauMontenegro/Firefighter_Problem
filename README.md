# DYNAFF (Dynamic Programming Algorithms Solving Fire Fighter Problem)

This Library contains a framework to solve a more restricted version for Fire Fighting Problem where one or more agents 
have some time-cost when try to reach each node in the Burning Graph. This limitation generates some nodes that are invalid
to travel due to fire reaching node in less time than the agent.

Some Features:
  - DP Solution Tested in Trees
  - DP Solution for Bulldozer CAENV: https://github.com/elbecerrasoto/gym-cellular-automata
  - DP Solution for Random Trees with Non-Metric Distances

On Progress:
  -Testing Multiple Fires  
  -Testing Other DP Algorithms
  
  
  How to execute?:
  python dynaff.py -s dpsolver_mau -i caenv -c rndtree_config
  
    -s is the algorithm we want to use
    -i is the env type in the input
    -c is the config file for solver and input type
    

