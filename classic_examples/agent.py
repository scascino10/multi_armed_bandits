import numpy as np
import random


def update_Qs(Qs_hat, a, reward, ns):
  Qs_hat[a] += (reward - Qs_hat[a]) / ns[a]


def random_action(Qs_hat):
  return np.random.choice(len(Qs_hat))


def greedy_action(Qs_hat):
  return np.argmax(Qs_hat)


def eps_greedy_action(Qs_hat, eps):
  if random.random() < eps:
    return random_action(Qs_hat)
  else:
    return greedy_action(Qs_hat)
