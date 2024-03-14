### 
import time
# will store tuples of ( state, action, reward, next_state, done )
import numpy as np

import wandb  # Make sure to import wandb

import gym
import random 
import os 
# Create the Cart-Pole environment
import torch
import random
### we sample from these tuples to train the model
##### model takes state and action and returns a value 

import torch
import pickle
import wandb  # Make sure to import wandb

# Initialize a new wandb run
wandb.init(project='deep_q_learning_project', entity='byyoung3')



from collections import namedtuple, deque
class cpmodel(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(cpmodel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = torch.nn.Linear(5, 500)
        self.fc2 = torch.nn.Linear(500, 128)
        self.fc3 = torch.nn.Linear(128, 1)

    def forward(self, x):
        # print(x.shape)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

env = gym.make('CartPole-v1')
model = cpmodel(5, 1)
e_model = cpmodel(5, 1)



# load model from pickle if its there
# model = pickle.load(open("model.pkl", "rb")) if os.path.exists("model.pkl") else model


# Your existing code to setup the environment and model


epsilon = 1.0  # Initial epsilon value (start with full exploration)
epsilon_min = 0.2  # Minimum epsilon value (residual exploration)
epsilon_decay_rate = 0.995  # Rate at which epsilon decays per iteration
update_every = 20  # Update target network every 20 episodes, adjust as needed
gamma = 0.99
replay_buffer = deque([], maxlen=10000)


def evaluate_model(model,env, n_episodes=1):
    
    total_scores = []  # To keep track of the total score for each episode

    for episode in range(n_episodes):
        state = env.reset()

        done = False
        total_reward = 0  # Total reward for this episode
        stps = 0 
        while not done:
            stps+=1 
            # Convert state into tensor for model input, omitting rendering for speed

            if isinstance(state, tuple):
                actual_state = state[0]
            else:
                actual_state = state    
            state = actual_state        
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            
            # Select action with the highest Q-value
            qs = []
            for action in [0, 1]:  # Assuming two possible actions
                state_action = torch.cat((state_tensor, torch.tensor([[action]], dtype=torch.float)), dim=1)
                q_value = model(state_action).item()
                qs.append(q_value)
            
            best_action = np.argmax(qs)
            
            # Take action in the environment
            next_state, reward, done, _, _ = env.step(best_action)
            total_reward += reward
            time.sleep(0.001)
            # Move to the next state
            state = next_state
            if stps > 200:
                # time.sleep(0.3)
                done = True
                break
        # Store the total reward for this episode
        total_scores.append(total_reward)
        


    # Calculate and print the average score
    average_score = sum(total_scores) / n_episodes
    print(f'Average Score over {n_episodes} episodes: {average_score}')

    return average_score  # Optionally return the average score



def optimize_model(policy_model, target_model ): 
    global replay_buffer
    samples = random.sample(replay_buffer,batch_size)

    # Organize data
    states = torch.tensor([sample[0] for sample in samples], dtype=torch.float)
    actions = torch.tensor([sample[1] for sample in samples], dtype=torch.long)
    rewards = torch.tensor([sample[2] for sample in samples], dtype=torch.float)
    next_states = torch.tensor([sample[3] for sample in samples], dtype=torch.float)
    dones = torch.tensor([sample[4] for sample in samples], dtype=torch.float)


    best_actions = torch.zeros(batch_size, dtype=torch.long)  # Assuming a batch size of 128

    # Iterate over all possible actions to find which ones yield the highest Q-values for each next state
    for a in [0, 1]:  # Assuming two possible actions
        # Convert action to tensor and repeat it for each sample in the batch
        action_tensor = torch.full((batch_size, 1), a, dtype=torch.float)
        
        # Concatenate the action tensor to the end of each next state tensor
        nsa = torch.cat((next_states, action_tensor), dim=1)
        
        # Use the online model (or target, but here we follow your setting) for prediction
        q_values = policy_model(nsa).squeeze()  # Remove unnecessary dimensions, assuming model outputs shape (batch, 1)
        
        # This checks if the current action 'a' is better than the previously recorded best action for each next state
        if a == 0:  # For the first action, initialize max_qs with the q_values directly
            max_qs = q_values
            best_actions.fill_(a)  # Fill with the current action 'a'
        else:  # For subsequent actions, update max_qs and best_actions wherever the new q_values exceed the current max_qs
            better_idx = q_values > max_qs  # Find indices where current q_values exceed max_qs
            max_qs[better_idx] = q_values[better_idx]  # Update max_qs
            best_actions[better_idx] = a  # Update the actions for these indices to the current action 'a'


    # Prepare the next state-action pairs using the best actions determined from the online model
    next_state_actions = best_actions.unsqueeze(1)  # Add a second dimension to best_actions
    nsa_pairs = torch.cat([next_states, next_state_actions.float()], dim=1)  # Concatenate along the second dimension

    # Evaluate these pairs using the target network (e_model) to get the Q-value estimates
    next_state_values = target_model(nsa_pairs).squeeze()  # Remove unnecessary dimensions

    # The (1 - dones) term ensures that we set the target Q-value to just the reward for terminal states
    targets = rewards + (gamma * next_state_values * (1 - dones))  # Assuming gamma (discount factor) is 0.99

    sa_pairs = torch.cat([states, actions.unsqueeze(1) ], dim=1)  # Concatenate states and actions

    predicted_q_values = policy_model(sa_pairs).squeeze()  # Ensure this matches the shape of targets

    loss = criterion(targets, predicted_q_values )

    # Perform the gradient descent step to update the online model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.item())



best_avg_score = float('-inf')  # Initialize the best average score to negative infinity

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = torch.nn.HuberLoss()
batch_size = 128
e_model.load_state_dict(model.state_dict())  # Make sure both models start off with the same weights

max_eps = 10000

for st in range(): 
    global_replay_buffer = []
    total_episodes = 0
    epsilon = max(epsilon_min, epsilon * epsilon_decay_rate)
    
    # Log the epsilon value to wandb
    wandb.log({'epsilon': epsilon, 'step': st})
    
    state = env.reset()
    done = False

    while not done:
        if isinstance(state, tuple):
            actual_state = state[0]
        else:
            actual_state = state

        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            vs = []
            for a in [0, 1]:
                inp = torch.tensor(list(actual_state) + [a], dtype=torch.float)
                vs.append(model(inp).item())
            action = torch.argmax(torch.tensor(vs)).item()

        next_state, reward, done, truncated, info = env.step(action)
        replay_buffer.extend([(actual_state, action, reward, next_state, done)])
        state = next_state

        if len(replay_buffer) >= 1000:
            loss = optimize_model(policy_model=model, target_model=e_model)
            avg_score = evaluate_model(model=model, env=env, n_episodes=10)
            wandb.log({'average_score': avg_score, 'step': st})

            if avg_score > best_avg_score:
                best_avg_score = avg_score
                # Save the model as the best one so far
                torch.save(model.state_dict(), 'best_model.pth')
                # Log the new best score to wandb
                wandb.log({'best_avg_score': best_avg_score, 'step': st})

            if st % 20 == 0:
                e_model.load_state_dict(model.state_dict())

    env.close()
