from agent import update_Qs, greedy_action, eps_greedy_action
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from multiprocessing import Pool

K = 10
RUNS = 2000
TRAIN_STEPS = 1000
PLOT_FILENAME = 'plot.png'


def agent_worker(method, envs, optimism=0):
  result = np.zeros(TRAIN_STEPS)
  for run in range(RUNS):
    env = envs[run]
    run_result = np.zeros(TRAIN_STEPS)
    Qs_hat = np.zeros(K) + optimism
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
  envs = np.random.normal(0, 1, (RUNS, K))

  methods = [(partial(eps_greedy_action, eps=0.1), envs, 0),
             (partial(eps_greedy_action, eps=0.01), envs, 0),
             (greedy_action, envs, 0),
             (partial(eps_greedy_action, eps=0.1), envs, 5),
             (partial(eps_greedy_action, eps=0.01), envs, 5),
             (greedy_action, envs, 5)]

  print('starting processes')
  with Pool() as pool:
    results = pool.starmap(agent_worker, methods)
  print('done')
  print('saving results')
  for result in results:
    plt.plot(result)
  plt.xlabel('step')
  plt.ylabel('average reward')
  plt.legend(['eps 0.1', 'eps 0.01', 'greedy',
              'eps 0.1 opt 5', 'eps 0.01 opt 5', 'greedy opt 5'])
  plt.savefig(PLOT_FILENAME, dpi=400)
  print(f'results saved in {PLOT_FILENAME}')


if __name__ == '__main__':
  main()
