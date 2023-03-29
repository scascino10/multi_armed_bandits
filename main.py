from agent import update_Qs, greedy_action, eps_greedy_action
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from multiprocessing import Pool

K = 10
RUNS = 2000
TRAIN_STEPS = 1000
PLOT_FILENAME = 'plot.png'


def agent_worker(method):
  result = np.zeros(TRAIN_STEPS)
  for run in range(RUNS):
    env = np.random.normal(0, 1, K)
    run_result = np.zeros(TRAIN_STEPS)
    Qs_hat = np.zeros(K)
    ns = np.zeros(K) + 1e-7
    for step in range(TRAIN_STEPS):
      a = method(Qs_hat)
      reward = np.random.normal(env[a], 1)
      update_Qs(Qs_hat, a, reward, ns)
      ns[a] += 1
      run_result[step] = reward
    result += (run_result - result) / (run + 1)
  return result


def main():
  methods = [partial(eps_greedy_action, eps=0.1),
             partial(eps_greedy_action, eps=0.01),
             greedy_action]

  print('starting processes')
  with Pool(processes=len(methods)) as pool:
    results = pool.map(agent_worker, methods)
  print('done')

  for result in results:
    plt.plot(result)
  print('saving results')
  plt.legend(['eps 0.1', 'eps 0.01', 'greedy'])
  plt.savefig(PLOT_FILENAME, dpi=400)
  print(f'results saved in {PLOT_FILENAME}')


if __name__ == '__main__':
  main()
