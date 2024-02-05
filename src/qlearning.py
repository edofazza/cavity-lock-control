import time
from src.redpitaya import RedPitayaController
import numpy as np
import u3


def round_to_nearest_0_1(value):
    return round(value * 10) / 10


def round_to_nearest_0_05(value):
    return round(value * 20) / 20


class RedPitayaQLearning(object):
    def __init__(self, hostname: str, user: str = 'root', password: str = 'root', config: str = 'fermi',
                 gui: bool = False, load=False, learning_rate=0.4, discout_factor=0.99, epsilon=0.7,
                 num_episodes=32, test=False):
        self.rpc = RedPitayaController(hostname, user, password, config, gui)
        # states
        self.voltage_range = np.arange(-1, 1.1, 0.1)
        self.num_states = len(self.voltage_range)  # 21
        # actions
        self.action_range = np.arange(-.001, .0015, .0005)
        self.num_actions = len(self.action_range)  # 5
        # Q-Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discout_factor
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.tmp35 = u3.U3()
        self.test = test

        if self.test:
            self.epsilon = 0

        # Q-values
        if load:
            print('Loaded Q-Matrix')
            self.Q = np.load('q_qlearning.npy')
        else:
            self.Q = np.zeros((self.num_states, self.num_actions))

    def _get_state_index(self, temperature):
        return np.argmin(np.abs(self.voltage_range - temperature))

    def _get_action_index(self, action):
        return np.argmin(np.abs(self.action_range - action))

    def qlearning(self, episode: int = 0):
        while episode < self.num_episodes:
            print(f'EPISODE {episode}')
            if episode == 1000 and not self.test:
                self.epsilon = 0.3
            self.rpc.reset()
            self.rpc.ramp_piezo()
            if self.rpc.scan_temperature(500):
                time.sleep(10)
                try:
                    self.rpc.lockCavity()
                except:
                    continue
                system_unlock = False
                print(f'\tLocked at: {time.time()}')
                print(f'\tInitial temperature voltage: {self.rpc.redpitaya.ams.dac2}V')
                purple_signal, _ = self.rpc.scope()
                print(f'\tFast signal mean: {purple_signal.max()}')
                state = self._get_state_index(round_to_nearest_0_1(purple_signal.max()))
                while True:
                    time.sleep(1)
                    # Choose action using epsilon-greedy policy
                    if np.random.rand() < self.epsilon and not self.test:
                        action = np.random.choice(self.num_actions)
                    else:
                        action = np.argmax(self.Q[state, :])
                    print(f'\tAction index: {action}')
                    self.rpc.setdac2(self.rpc.redpitaya.ams.dac2 + self.action_range[action])
                    print(f'\tTemperature voltage: {self.rpc.redpitaya.ams.dac2}V')
                    time.sleep(0.1)
                    # Get the next state, reward, and system_unlock
                    purple_signal, blue_signal = self.rpc.scope()
                    print(f'\tFast signal mean: {purple_signal.max()}')
                    if blue_signal.max() < 0.95:
                        print(f'\tLost lock at: {time.time()}')
                        system_unlock = True
                    next_state = self._get_state_index(round_to_nearest_0_1(purple_signal.max()))
                    reward = 1 if not system_unlock else 0
                    print(f'\tState index: {next_state}')
                    ain0bits, = self.tmp35.getFeedback(u3.AIN(0))
                    ain0Value = self.tmp35.binaryToCalibratedAnalogVoltage(ain0bits,
                                                                           isLowVoltage=False,
                                                                           channelNumber=0)
                    ain2bits, = self.tmp35.getFeedback(u3.AIN(2))
                    ain2Value = self.tmp35.binaryToCalibratedAnalogVoltage(ain2bits,
                                                                           isLowVoltage=False,
                                                                           channelNumber=0)
                    print(f'\tTMP35 AIN0: {ain0Value}')
                    print(f'\tTMP35 AIN2: {ain2Value}')
                    # Update Q-values
                    if not self.test:
                        max_next_q = np.max(self.Q[next_state, :])
                        self.Q[state, action] = (self.Q[state, action] +
                                                 self.learning_rate * (reward +
                                                                       self.discount_factor * max_next_q -
                                                                       self.Q[state, action]))
                    state = next_state

                    if system_unlock:
                        if not self.test:
                            np.save('q_qlearning.npy', self.Q)
                        episode += 1
                        break


if __name__ == '__main__':
    rpql = RedPitayaQLearning('169.254.167.128', load=True, num_episodes=5)
    rpql.qlearning()
