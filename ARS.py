# Augmented Random Search 
#pip install gym==0.10.5
#pip install pybullet==2.0.8
# and install " conda install -c conda-forge ffmpeg " for video output
#only for windows users download and install Visual C++ Build Tools from 
# https://visualstudio.microsoft.com/visual-cpp-build-tools/  so gym can work
import os
import numpy as np
import gym 
from gym import wrappers
import pybullet_envs 
# hyper parameters has fixed values 
# Hp => hyper parameters
class Hp():
    def __init__(self):
        # instances 
        self.nb_steps = 1000 #number of training lopps we are going to have at the end  or number of times to update the model
        self.episode_length = 1000 # maximum of time to let the AI to try to walk
        self.learning_rate = 0.02 #how fast is the AI learning
        self.nb_directions = 16 #the bigger the number the longer it takes to learn
        self.nb_best_directions = 16 
        assert self.nb_best_directions <= self.nb_directions 
        self.noise = 0.03 # Gaussian
        self.seed = 1
        #self.env_name = 'HumanoidBulletEnv-v0'
        self.env_name = 'HalfCheetahBulletEnv-v0'
        
# normalizing this bad boy so it has a good performance
class Normalizer():
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)  #total number of states
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs) #the numerator
        self.var = np.zeros(nb_inputs) #the variance

#update and compute the mean and the variance whenever a state is observed
    def observe(self, x): #x is a new state        
        self.n += 1. 
        last_mean = self.mean.copy() 
        self.mean += (x-self.mean) / self.n #the mean....
        self.mean_diff += (x - last_mean) * (x - self.mean)  #the variance
        self.var = (self.mean_diff / self.n).clip(min =1e-2) #0.01
    def normalize(self, inputs):
        obs_mean = self.mean #the observed mean
        obs_std = np.sqrt(self.var) #the observe standard deviation
        return (inputs - obs_mean) / obs_std  
    
#building the AI starts here
#the AI is a policy
class Policy():
    
    def __init__(self, input_size, output_size):
        self.theta = np.zeros((output_size, input_size))     
                            
    def evaluate(self, input, delta = None, direction = None):
        if direction is None:
            return self.theta.dot(input)
        elif direction == "positive":
            return (self.theta + hp.noise * delta).dot(input)
        else: 
            return (self.theta - hp.noise * delta).dot(input)
        
        #sampling the deltas
    def sample_deltas (self): #gausian distribution of mean 0 and variance 1
         return [np.random.randn(*self.theta.shape) for _ in range(hp.nb_directions)]
     
    def update(self, rollout,  sigma_r):
         step = np.zeros(self.theta.shape)
         for r_pos, r_neg, d in rollout:
             step += (r_pos - r_neg) * d
         self.theta += hp.learning_rate /  (hp.nb_best_directions * sigma_r) * step

# exploring the policy on one specific direction and over one episode    

def explore (env, Normalizer, policy, direction = None, delta = None):
    state = env.reset() 
    done = False
    num_plays = 0. 
    sum_rewards = 0
    while not done and num_plays < hp.episode_length:
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        state, reward, done, _ = env.step(action)
        reward = max(min(reward, 1), -1) #to prevent bias
        sum_rewards += reward
        num_plays += 1
    return sum_rewards

#Training the AI

def train(env, policy, normalizer, hp):
    for step in range(hp.nb_steps):
        
        #initializing the pertubations deltas & the positive/negative rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.nb_directions
        negative_rewards = [0] * hp.nb_directions
        
        #getting the positive rewards in the positive directions
        for k in range(hp.nb_directions):
            positive_rewards[k] = explore(env, normalizer, policy, direction = "positive", delta = deltas[k])
        
        #getting the negative rewards in the negative/positive directions
        for k in range(hp.nb_directions):
            negative_rewards[k] = explore(env, normalizer, policy, direction = "negative", delta = deltas[k])
            
        #gathering all the positive/negative rewards to compute the standard deviation of these rewards
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()
        
        #sorting the rollouts by the max(r_pos, r_neg) and selecting the best direction
        scores = {k:max(r_pos, r_neg) for k,(r_pos,r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
        order = sorted(scores.keys(), key = lambda x:scores[x], reverse = True)[:hp.nb_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
        
         # Updating our policy
        policy.update(rollouts, sigma_r)
        
        # Printing the final reward of the policy after the update
        reward_evaluation = explore(env, normalizer, policy)
        print('Step:', step, 'Reward:', reward_evaluation)
      
# Running the main code
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')

hp = Hp()
np.random.seed(hp.seed)
env = gym.make(hp.env_name)
env = wrappers.Monitor(env, monitor_dir, force = True)
nb_inputs = env.observation_space.shape[0]
nb_outputs = env.action_space.shape[0]
policy = Policy(nb_inputs, nb_outputs)
normalizer = Normalizer(nb_inputs)
train(env, policy, normalizer, hp)