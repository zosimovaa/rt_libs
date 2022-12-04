# Reinforcement learning train/trade core
## Description

rt_libs - a set of components for implementing a neural network training environment and trading on the stock exchange.

In general, the process looks like this:
1. Poloniex Data Collector collects data from the exchange into the Clickhouse database - the state of orderbooks (asks and bids,) and trades history.
2. The DQN algorithm trains the neural network in the environment to achieve the maximum reward. The logic of the environment is implemented in the core component rt_libs
3. The trained model is launched in the trader and makes a decision online. The core component from rt_libs is also involved here
4. Trading results are stored in the Clickhouse database


## Installation
``pip install git+https://zosimovaa@bitbucket.org/zosimovaa/rt_core.git``

## Contacts
- **e-mail**: lesha.spb@gmail.com
- **telegram**: https://t.me/lesha_spb


