import gym
import torch
import matplotlib.pyplot as plt
from IPython import display

env = gym.make('CartPole-v1',render_mode='human')


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

model = cpmodel(4, 1)  # Assuming state size of 4 for CartPole
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

state = env.reset()
# img = plt.imshow(env.render())  # Only for initialization
done = False

while not done:
    display.clear_output(wait=True)
    img = env.render()
    # img.set_data()  # Update the data
    # plt.axis('off')
    # display.display(plt.gcf())

    actual_state = state if not isinstance(state, tuple) else state[0]

    vs = []
    for a in range(env.action_space.n):
        inp = torch.tensor(list(actual_state) + [a], dtype=torch.float)
        vs.append(model(inp).item())
    action = torch.argmax(torch.tensor(vs)).item()

    state, reward, done, truncated, info = env.step(action)

env.close()
