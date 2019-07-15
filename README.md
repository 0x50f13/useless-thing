# useless-thing
Useless try to create neural network.


# The idea
Instead of using gradient descent for learning my neural network I've tried simplier method. I tried to do the following:<br>
` layer[j]=layer[j]+activity[i]*sign(layer[j])*reward*lr ` <br>
where is weight of jth input of each neuron. activity is latest value of that input. i is number of input. So it increase the absolute value of weight if action was rewarded and decrease it if not. But unfortanetely it doesn't work.

# Testing it
This network was tested on LunarLanderContinous-v2 from gym library. On average it do 1 landing over 1000 iterations.

# Running 
` python rl1.py ` <br>
**NOTICE**: ` gym ` library is required to run. 
