import time
import logging
import torch
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from collections import defaultdict
from controlling.train import Agent  # Assuming this exists in ./controlling
import optparse

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

def flip_bit(a):
    """Toggle between 0 and 1 to switch traffic light direction."""
    return 1 if a == 0 else 0

def countCarState(fn, max_frame, state_dict):
    """Load car state data and prepare state_dict with vehicle counts per lane over time."""
    state = pd.read_csv(fn, header=None, sep=' ')
    dict_input = {state.iloc[row, 1]: [state.iloc[row, 2] - 1, state.iloc[row, 3]] for row in range(state.shape[0])}

    # Initial empty state
    lst = [0] * 12
    state_dict.append(lst)

    for frame in range(max_frame):
        row = dict_input.get(frame, [0, 0])
        current_state = lst.copy()
        
        if row != [0, 0]:  # Update only if there's new data
            current_state[row[0]] += row[1]
        
        lst = current_state.copy()
        state_dict.append(current_state)

class Model(nn.Module):
    """Neural network model for the traffic light decision-making."""
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dims, fc1_dims)
        self.linear2 = nn.Linear(fc1_dims, fc2_dims)
        self.linear3 = nn.Linear(fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        actions = self.linear3(x)
        return actions

def run_process(args):
    """Run the traffic light control simulation with detailed logging."""
    # Only one junction for simplicity
    all_junctions = [0]
    logger.info("Initializing agent and model...")
    brain = Agent(
        gamma=0.99,
        epsilon=0.0,
        lr=0.1,
        input_dims=12,
        fc1_dims=256,
        fc2_dims=256,
        batch_size=1024,
        n_actions=10,
        junctions=all_junctions,
        Q_eval=Model(0.1, 12, 256, 256, 10)
    )

    # Load pre-trained model weights
    brain.Q_eval.load_state_dict(torch.load(args.model_path, map_location=brain.Q_eval.device, weights_only=True))

    # Initialize parameters for traffic light control
    step = 1
    min_duration = 3
    cur_direction = 0  # Traffic light direction (0 or 1)
    traffic_lights_time = defaultdict(lambda: 0)  # Timer for each traffic light
    state_dict = []
    countCarState(args.state_path, args.n_steps, state_dict)

    last_step = 0

    # Start the simulation
    logger.info("Starting traffic light control simulation...")
    while step <= min(args.n_steps, len(state_dict)):
        for junction in all_junctions:
            if step > 150:  # Start controlling after 150 steps
                if traffic_lights_time[junction] <= 0:
                    # Calculate the vehicle count difference to determine the state
                    vehicles_per_lane = [a_i - b_i for a_i, b_i in zip(state_dict[step], state_dict[last_step])]
                    logger.info(f"Step {step}: Vehicles per lane {vehicles_per_lane}")
                    
                    # Get action (phase time) from the model based on the current state
                    phase_time = brain.choose_action(vehicles_per_lane)
                    traffic_lights_time[junction] = min_duration + phase_time
                    logger.info(f"Direction: {cur_direction}, Duration: {traffic_lights_time[junction]} seconds")

                    # Update traffic light direction for next phase
                    cur_direction = flip_bit(cur_direction)
                    last_step = step
                else:
                    traffic_lights_time[junction] -= 1 / 15  # Decrement timer in fractional seconds
                    logger.info(f"Continuing in Direction {cur_direction}, Remaining Time: {traffic_lights_time[junction]:.2f} seconds")

        step += 1
        time.sleep(1 / 15)  # Simulation time step

def get_options():
    """Parse command-line options for model path, steps, and state file."""
    optParser = optparse.OptionParser()
    optParser.add_option('--model_path', type='string', default='./controlling/models/1st_test.bin', help='Path to the model file')
    optParser.add_option('--n_steps', type='int', default=300, help='Number of steps for simulation')
    optParser.add_option('--state_path', type='string', default='./detection/experiments/cam4/output.txt', help='Path to input state file')
    options, args = optParser.parse_args()
    return options

if __name__ == "__main__":
    options = get_options()
    run_process(options)
