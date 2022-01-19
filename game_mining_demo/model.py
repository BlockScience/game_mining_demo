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
                   actions: list[int]):
        """
        
        actions: {time_no: action_no}
        """

        if len(actions) > 1:
            alpha_weight = 5
            beta_weight = 2
            gamma_weight = 1
            total_weight = alpha_weight + beta_weight + gamma_weight
            alpha = alpha_weight / total_weight
            beta = beta_weight / total_weight
            gamma = gamma_weight / total_weight

            N_i = 10
            N_b = 5

            def estimator(lst):
                return sum(lst) / len(lst)

            proportional = estimator(actions)

            past_estimators = []
            if len(actions) > (N_i + N_b + 2):
                for i in range(N_b):
                    n = len(actions)
                    eff_actions = actions[n-(N_i + i):(n - i)]
                    past_estimators.append(estimator(eff_actions))
                integral = sum(past_estimators) / len(past_estimators)
                derivative = (past_estimators[0] - past_estimators[1]) / 2
            else:
                integral = proportional
                derivative = proportional


            return alpha * proportional + beta * integral + gamma * derivative
        else:
            return 1 / 2


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
    'past_actions': {0: [], 1: []},
    'payoffs': {},
    'payoff_tensor': PAYOFF_TENSOR
}


def s_actions(params, _2, _3, state, _5):

    past_actions = state['past_actions']
    players: list[Player] = state['players']
    payoff_tensor: list[list[list[int]]] = state['payoff_tensor']

    choices = {}

    for i, player in enumerate(players):

        if player.player_no == 0:
            other_player_actions = past_actions[1]
        elif player.player_no == 1:
            other_player_actions = past_actions[0]
        else:
            raise Exception()

        p_1 = player.likelihood(other_player_actions)

        # Payoff matrix and probability vector
        R = payoff_tensor[i]
        P = (p_1, 1 - p_1)

        # Compute expected payoffs
        if player.player_no == 0:
            pi_1 = R[0][0] * P[0] + R[0][1] * P[1]
            pi_2 = R[1][0] * P[0] + R[1][1] * P[1]
        elif player.player_no == 1:
            pi_1 = R[0][0] * P[0] + R[1][0] * P[1]
            pi_2 = R[0][1] * P[0] + R[1][1] * P[1]
        else:
            pass
    
        # Iverson Bracket
        if pi_1 > pi_2:
            choice = 1
        else:
            choice = 0

        choices[player.player_no] = choice

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
    if state['timestep'] > 10:
        
        sigma = np.random.randn(2, 2, 2)
        new_T = T + sigma
        return ('payoff_tensor', new_T.tolist())
    else:
        return ('payoff_tensor', T.tolist())


def s_past_actions(_1, _2, _3, state, _5):
    h = state['past_actions'].copy()
    a = state['actions']

    h[0].append(a[0])
    h[1].append(a[1])
    return ('past_actions', h)


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
        'name': 'Past Actions',
        'policies': {

        },
        'variables': {
            'past_actions': s_past_actions
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
