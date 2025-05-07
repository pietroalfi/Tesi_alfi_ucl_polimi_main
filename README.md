# Deep Duelling Double Q-learning for Limit Order Book Trading

See the [wiki](https://github.com/florisdobber/ucl-thesis/wiki) for information about the project, meeting notes and other related material. 

## Getting Started
### Installation

#### Installing Python
1. Install Miniconda using [these instructions](https://docs.anaconda.com/miniconda/)

2. Create a new Conda environment `conda create --prefix /choose/your/location/ENVNAME python=3.9`

3. Activate the environment `conda activate /choose/your/location/ENVNAME`

#### Installing the repository

1. Download the ABIDES source code, either directly from GitHub or with git:

```bash
git clone https://github.com/florisdobber/ucl-thesis
```

2. Run the install script to install the ABIDES packages and their dependencies:

```
sh install.sh
```

## About The Project

ABIDES (Agent Based Interactive Discrete Event Simulator) is a general purpose multi-agent discrete event simulator. Agents exclusively communicate through an advanced messaging system that supports latency models.

The project is currently broken down into 3 parts: ABIDES-Core, ABIDES-Markets and ABIDES-Gym.

* ABIDES-Core: Core general purpose simulator that be used as a base to build simulations of various systems.
* ABIDES-Markets: Extension of ABIDES-Core to financial markets. Contains implementation of an exchange mimicking NASDAQ, stylised trading agents and configurations.
* ABIDES-Gym: Extra layer to wrap the simulator into an OpenAI Gym environment for reinforcement learning use. 2 ready to use trading environments available. Possibility to build other financial markets environments easily.