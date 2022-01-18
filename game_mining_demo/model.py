from typing import Annotated
from dataclasses import dataclass
import numpy as np
import xarray as xr
from random import random

TIMESTEPS = 1000
SAMPLES = 1

PLAYER_1_PAYOFF = [[1, 0.5], [0.5, 0.1]]
PLAYER_2_PAYOFF = [[1, 0.5], [0.5, 0.1]]
PAYOFF_TENSOR = [PLAYER_1_PAYOFF, PLAYER_2_PAYOFF]


@dataclass
class Player():
    actions_belief: Annotated[dict['Player', list], None]
    player_no: int

    def likelihood(self,
                   actions: dict[int, dict[int, int]]):
        alpha_weight = 5
        beta_weight = 2
        gamma_weight = 1
        total_weight = alpha_weight + beta_weight + gamma_weight
        alpha = alpha_weight / total_weight
        beta = beta_weight / total_weight
        gamma = gamma_weight / total_weight

        non_player_actions = {k: v
                              for k, v
                              in actions.items()
                              if k != self.player_no}

        # TODO
        proportional = 0.0
        integral = proportional
        derivative = proportional

        return alpha * proportional + beta * integral + gamma * derivative


# %%


INITIAL_PLAYERS = [
    Player(actions_belief={}, player_no=0),
    Player(actions_belief={}, player_no=1)
]

PARAMS: dict = {
    'nothing': [0]
}

INITIAL_STATE: dict = {
    'players': INITIAL_PLAYERS,
    'actions': {},
    'payoffs': {},
    'payoff_tensor': PAYOFF_TENSOR
}


def s_actions(params, _2, _3, state, _5):

    players: list[Player] = state['players']
    payoff_tensor: list[list[list[int]]] = state['payoff_tensor']

    choices = {}
    for i, player in enumerate(players):
        payoff = payoff_tensor[i]
        p_a1 = player.likelihood(state['actions'])
        p_a2 = 1 - p_a1

        payoff_1_estimator = p_a1 * payoff[0][0] + p_a2 * payoff[0][1]
        payoff_2_estimator = p_a1 * payoff[1][0] + p_a2 * payoff[1][1]
        # payoff_avg_estimator = (payoff_1_estimator + payoff_2_estimator) / 2
        # payoff_min = min(payoff_1_estimator, payoff_2_estimator)
        # payoff_max = max(payoff_1_estimator, payoff_2_estimator)
        utility_1 = payoff_1_estimator
        utility_2 = payoff_2_estimator
        total_utility = sum([utility_1, utility_2])

        probability_1 = utility_1 / total_utility
        choice = (random() > probability_1)
        choices[i] = choice

    return ('actions', choices)


def s_payoffs(params, _2, _3, state, _5):

    actions: dict[int, bool] = state['actions']
    payoff_tensor: list[list[list[int]]] = state['payoff_tensor']
    players: list[Player] = state['players']

    payoffs = {i: payoff_tensor[i][actions[0]][actions[1]]
               for i, player in enumerate(players)}

    return ('payoffs', payoffs)


def s_players(params, _2, _3, state, _5):
    return ('players', state['players'])



def s_payoff_tensor(params, _2, _3, state, _5):

    T = np.array(state['payoff_tensor'])
    if state['timestep'] > 100:
        
        sigma = np.random.randn(2, 2, 2)
        new_T = T + sigma
        return ('payoff_tensor', new_T.tolist())
    else:
        return ('payoff_tensor', T.tolist())



BLOCKS: list[dict] = [
    {
        'name': 'Play the game',
        'policies': {

        },
        'variables': {
            'actions': s_actions
        }
    },
    {
        'name': 'Compute payoffs',
        'ignore': False,
        'policies': {

        },
        'variables': {
            'payoffs': s_payoffs
        }
    },
    {
        'name': 'Update beliefs',
        'ignore': False,
        'policies': {
        },
        'variables': {
            'players': s_players

        }
    },
    {
        'name': 'Mutate payoff tensor',
        'ignore': False,
        'policies': {

        },
        'variables': {
            'payoff_tensor': s_payoff_tensor

        }
    }
]

BLOCKS = [block
          for block in BLOCKS
          if block.get('ignore', False) == False]
