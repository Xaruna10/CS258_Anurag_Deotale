import networkx as nx
import gym
from gym import spaces
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
import matplotlib.pyplot as plt
from ray.rllib.env.env_context import EnvContext

# Load network topology from a GML file
def load_network_topology(file_path):
    return nx.read_gml(file_path)

network = load_network_topology('nsfnet.gml')
num_nodes = len(network.nodes)

# Define custom gym environment for optical networks
class OpticalNetworkEnv(gym.Env):
    def __init__(self, config: EnvContext):
        super(OpticalNetworkEnv, self).__init__()
        self.network = config["network"]
        self.num_requests = config["num_requests"]
        self.min_ht = config["min_ht"]
        self.max_ht = config["max_ht"]
        self.case = config.get("case", "II")
        self.capacity = {tuple(sorted(edge)): 10 for edge in self.network.edges}
        self.node_to_index = {node: idx for idx, node in enumerate(self.network.nodes)}
        self.index_to_node = {idx: node for node, idx in self.node_to_index.items()}

        self.action_space = spaces.Discrete(num_nodes)
        self.observation_space = spaces.Discrete(num_nodes)

        self.current_step = 0
        self.requests = self._generate_requests()
        self.current_request = self.requests[self.current_step]
        self.occupied_slots = {tuple(sorted(edge)): [] for edge in self.network.edges}
        self.total_occupied = {tuple(sorted(edge)): 0 for edge in self.network.edges}
        self.utilizations = []

    def _generate_requests(self):
        requests = []
        for _ in range(self.num_requests):
            if self.case == "I":
                source = self.node_to_index["San Diego Supercomputer Center"]
                destination = self.node_to_index["Jon Von Neumann Center, Princeton, NJ"]
            else:
                source = np.random.choice(list(self.node_to_index.values()))
                destination = np.random.choice(list(self.node_to_index.values()))
                while destination == source:
                    destination = np.random.choice(list(self.node_to_index.values()))
            holding_time = np.random.randint(self.min_ht, self.max_ht + 1)
            requests.append((source, destination, holding_time))
        return requests

    def reset(self):
        self.current_step = 0
        self.requests = self._generate_requests()
        self.current_request = self.requests[self.current_step]
        self.occupied_slots = {tuple(sorted(edge)): [] for edge in self.network.edges}
        self.total_occupied = {tuple(sorted(edge)): 0 for edge in self.network.edges}
        self.utilizations = []
        return self.current_request[0]

    def step(self, action):
        done = False
        reward = 0

        source, destination, holding_time = self.current_request
        if action == destination:
            path = nx.shortest_path(self.network, source=self.index_to_node[source], target=self.index_to_node[destination])
            success = self.allocate_spectrum(path, holding_time)
            reward = 1 if success else -1
        else:
            reward = -1

        self.current_step += 1
        if self.current_step >= self.num_requests:
            done = True
        else:
            self.current_request = self.requests[self.current_step]

        self.release_spectrum()
        self.utilizations.append(self.get_network_utilization())

        return self.current_request[0], reward, done, {}

    def allocate_spectrum(self, path, holding_time):
        try:
            for edge in zip(path[:-1], path[1:]):
                edge = tuple(sorted(edge))
                if len(self.occupied_slots[edge]) >= self.capacity[edge]:
                    return False
            for edge in zip(path[:-1], path[1:]):
                edge = tuple(sorted(edge))
                self.occupied_slots[edge].append(holding_time)
                self.total_occupied[edge] += 1
            return True
        except KeyError as e:
            print(f"KeyError: {e}, path: {path}, edge: {edge}")
            return False

    def release_spectrum(self):
        for edge in self.occupied_slots:
            self.occupied_slots[edge] = [ht - 1 for ht in self.occupied_slots[edge] if ht > 1]

    def get_network_utilization(self):
        total_utilization = 0
        for edge in self.network.edges:
            edge = tuple(sorted(edge))
            total_utilization += self.total_occupied[edge] / (self.capacity[edge] * self.num_requests)
        return total_utilization / len(self.network.edges)

    def render(self, mode="human"):
        pass

    def close(self):
        pass

# Function to configure and create trainers
def create_trainer(algorithm, config):
    if algorithm == "PPO":
        trainer_config = PPOConfig().environment(
            env=OpticalNetworkEnv,
            env_config=config
        ).rollouts(num_rollout_workers=1, rollout_fragment_length=50)
    elif algorithm == "DQN":
        trainer_config = DQNConfig().environment(
            env=OpticalNetworkEnv,
            env_config=config
        ).rollouts(num_rollout_workers=1, rollout_fragment_length=50)
    else:
        raise ValueError("Unsupported algorithm")
    return trainer_config.build()

# Initialize environment configurations
config_case_I = {
    "network": network,
    "num_requests": 100,
    "min_ht": 10,
    "max_ht": 20,
    "case": "I"
}

config_case_II = {
    "network": network,
    "num_requests": 100,
    "min_ht": 10,
    "max_ht": 20,
    "case": "II"
}

# Create trainers for Case I
ppo_trainer_case_I = create_trainer("PPO", config_case_I)
dqn_trainer_case_I = create_trainer("DQN", config_case_I)

# Create trainers for Case II
ppo_trainer_case_II = create_trainer("PPO", config_case_II)
dqn_trainer_case_II = create_trainer("DQN", config_case_II)

# Training function
def train(trainer, num_iterations=100):
    results = []
    utilizations = []
    for i in range(num_iterations):
        result = trainer.train()
        results.append(result)
        env = trainer.workers.local_worker().env
        if isinstance(env, OpticalNetworkEnv):
            utilization = env.get_network_utilization()
            utilizations.append(utilization)
            print(f"Iteration {i}: reward {result['episode_reward_mean']}, utilization {utilization}")
        else:
            print(f"Iteration {i}: reward {result['episode_reward_mean']}, utilization not available")
    return results, utilizations

# Train the trainers for both cases
results_ppo_case_I, utilizations_ppo_case_I = train(ppo_trainer_case_I)
results_dqn_case_I, utilizations_dqn_case_I = train(dqn_trainer_case_I)
results_ppo_case_II, utilizations_ppo_case_II = train(ppo_trainer_case_II)
results_dqn_case_II, utilizations_dqn_case_II = train(dqn_trainer_case_II)

# Evaluation function
def evaluate(env, trainer, num_episodes):
    rewards = []
    utilizations = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = trainer.compute_single_action(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        utilizations.append(env.get_network_utilization())
    return rewards, utilizations

# Evaluate PPO and DQN for both cases
env_case_I = OpticalNetworkEnv(config_case_I)
env_case_II = OpticalNetworkEnv(config_case_II)

ppo_rewards_case_I, ppo_utilizations_case_I_eval = evaluate(env_case_I, ppo_trainer_case_I, 100)
dqn_rewards_case_I, dqn_utilizations_case_I_eval = evaluate(env_case_I, dqn_trainer_case_I, 100)
ppo_rewards_case_II, ppo_utilizations_case_II_eval = evaluate(env_case_II, ppo_trainer_case_II, 100)
dqn_rewards_case_II, dqn_utilizations_case_II_eval = evaluate(env_case_II, dqn_trainer_case_II, 100)

# Implement heuristic algorithm for comparison
class HeuristicSpectrumAllocator:
    def __init__(self, network):
        self.network = network
        self.capacity = {tuple(sorted(edge)): 10 for edge in self.network.edges}
        self.occupied_slots = {tuple(sorted(edge)): [] for edge in self.network.edges}
        self.total_occupied = {tuple(sorted(edge)): 0 for edge in self.network.edges}

    def allocate(self, path, holding_time):
        try:
            for edge in zip(path[:-1], path[1:]):
                edge = tuple(sorted(edge))
                if len(self.occupied_slots[edge]) >= self.capacity[edge]:
                    return False
            for edge in zip(path[:-1], path[1:]):
                edge = tuple(sorted(edge))
                self.occupied_slots[edge].append(holding_time)
                self.total_occupied[edge] += 1
            return True
        except KeyError as e:
            print(f"KeyError: {e}, path: {path}, edge: {edge}")
            return False

    def release(self):
        for edge in self.occupied_slots:
            self.occupied_slots[edge] = [ht - 1 for ht in self.occupied_slots[edge] if ht > 1]

    def run_simulation(self, requests):
        rewards = []
        utilizations = []
        for request in requests:
            source, destination, holding_time = request
            path = nx.shortest_path(self.network, source=self.index_to_node[source], target=self.index_to_node[destination])
            if self.allocate(path, holding_time):
                rewards.append(1)
            else:
                rewards.append(-1)
            self.release()
            total_utilization = 0
            for edge in self.network.edges:
                edge = tuple(sorted(edge))
                total_utilization += self.total_occupied[edge] / (self.capacity[edge] * len(requests))
            utilizations.append(total_utilization / len(self.network.edges))
        return rewards, utilizations

# Run heuristic for both cases
heuristic_allocator_case_I = HeuristicSpectrumAllocator(network)
heuristic_allocator_case_II = HeuristicSpectrumAllocator(network)

requests_case_I = OpticalNetworkEnv(config_case_I)._generate_requests()
requests_case_II = OpticalNetworkEnv(config_case_II)._generate_requests()

heuristic_rewards_case_I, heuristic_utilizations_case_I = heuristic_allocator_case_I.run_simulation(requests_case_I)
heuristic_rewards_case_II, heuristic_utilizations_case_II = heuristic_allocator_case_II.run_simulation(requests_case_II)

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(ppo_rewards_case_I, label='PPO Case I')
plt.plot(dqn_rewards_case_I, label='DQN Case I')
plt.plot(heuristic_rewards_case_I, label='Heuristic Case I')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(ppo_rewards_case_II, label='PPO Case II')
plt.plot(dqn_rewards_case_II, label='DQN Case II')
plt.plot(heuristic_rewards_case_II, label='Heuristic Case II')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(utilizations_ppo_case_I, label='PPO Utilization Case I')
plt.plot(utilizations_dqn_case_I, label='DQN Utilization Case I')
plt.plot(heuristic_utilizations_case_I, label='Heuristic Utilization Case I')
plt.plot(utilizations_ppo_case_II, label='PPO Utilization Case II')
plt.plot(utilizations_dqn_case_II, label='DQN Utilization Case II')
plt.plot(heuristic_utilizations_case_II, label='Heuristic Utilization Case II')
plt.xlabel('Episode')
plt.ylabel('Network-wide Utilization')
plt.legend()

plt.tight_layout()
plt.show()
