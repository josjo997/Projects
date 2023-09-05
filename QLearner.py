
import numpy as np
import random as rand


class QLearner(object):

    def __init__(self, \
                 num_states=100, \
                 num_actions=4, \
                 alpha=0.2, \
                 gamma=0.9, \
                 rar=0.5, \
                 radr=0.99, \
                 dyna=0, \
                 verbose=False):

        self.verbose = verbose
        self.num_state = num_states
        self.num_actions = num_actions
        self.s = 0
        self.a = 0

        # initialize hypermeters and the Q table
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.q = np.zeros((num_states, num_actions))

        if dyna > 0:
            # debug
            self.seen = []

            self.t = np.full((num_states, num_actions, num_states), 0.00001)
            self.r = np.zeros((num_states, num_actions))
            self.t_optimization = {}
            self.history = {}

    def querysetstate(self, s):
        # update the state
        self.s = s

        # choose the next action
        random = rand.random()
        if random < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.q[s])

        return action

    def query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """

        # update the Q table
        self.q[self.s, self.a] = (1 - self.alpha) * self.q[self.s, self.a] + \
                                 self.alpha * (r + self.gamma * np.max(self.q[s_prime]))

        # dyna
        if self.dyna > 0:
            self.history[self.s] = list(set().union(self.history.get(self.s, []), [self.a]))

            # update T and R table
            self.t[self.s, self.a, s_prime] += 1
            self.r[self.s, self.a] = (1 - self.alpha) * self.r[self.s, self.a] + self.alpha * r

            # experience
            self.t_optimization = {}
            for _ in range(self.dyna):
                self.hal()

        # choose the next action
        random = rand.random()
        if random < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.q[s_prime])

        # update random action rate
        self.rar = self.rar * self.radr

        if self.verbose:
            print(
                "last state: {}, last action: {}, new state: {}, reward: {}, action: {}".format(self.s, self.a, s_prime,
                                                                                                r, action))

        # update the state and action
        self.s = s_prime
        self.a = action

        return action

    def hal(self):

        # randomly choose a state and an action
        s = rand.choice(list(self.history.keys()))
        a = rand.choice(self.history[s])

        # query into T for s_prime


        # real T probability approach
        # s_prime = self.choice(range(self.num_state), self.t[s,a]/np.sum(self.t[s,a]))

        # faster approach
        if (s, a) in self.t_optimization:
            s_prime = self.t_optimization[(s, a)]
        else:
            s_prime = np.argmax(self.t[s, a])
            self.t_optimization[(s, a)] = s_prime

        # query into R for r
        r = self.r[s, a]

        # update Q
        self.q[s, a] = (1 - self.alpha) * self.q[s, a] + \
                       self.alpha * (r + self.gamma * np.max(self.q[s_prime]))

        # print(self.q)

    def choice(self, options, probs):
        threshold = rand.random()
        n = 0
        for i, p in enumerate(probs):
            n += p
            if n > threshold:
                break
        return options[i]

if __name__ == "__main__":
    print("q-learn algo")