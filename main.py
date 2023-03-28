from agent import *
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm

K = 10
RUNS = 2000
TRAIN_STEPS = 1000


def main():
  methods = [partial(eps_greedy_action, eps=0.1),
             partial(eps_greedy_action, eps=0.01),
             greedy_action]

  results = np.zeros((len(methods), TRAIN_STEPS))
  for run in tqdm(range(RUNS)):
    env = np.random.normal(0, 1, K)
    for i, method in enumerate(methods):
      run_results = np.zeros(TRAIN_STEPS)
      Qs_hat = np.zeros(K)
      ns = np.zeros(K) + 1e-7
      for step in range(TRAIN_STEPS):
        a = method(Qs_hat)
        reward = np.random.normal(env[a], 1)
        update_Qs(Qs_hat, a, reward, ns)
        ns[a] += 1
        run_results[step] = reward
      results[i] += (run_results - results[i]) / (run + 1)

  for result in results:
    plt.plot(result)

  plt.legend(['eps 0.1', 'eps 0.01', 'greedy'])
  plt.savefig('plot.png', dpi=800)


if __name__ == '__main__':
  main()
