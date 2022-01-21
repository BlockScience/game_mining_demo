from game_mining_demo.types import Player
import numpy as np

def p_actions(params, _2, _3, state):

    past_actions = state['past_actions']
    players: list[Player] = state['players']
    payoff_tensor: list[list[list[int]]] = state['payoff_tensor']

    choices = {}
    likelihoods = {}
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
        likelihoods[player.player_no] = p_1

    return {'actions': choices,
            'likelihood': likelihoods}


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
    if (state['timestep'] % params['payoff_shift_interval']) == 0:

        st = np.random.RandomState()
        sigma = st.randn(2, 2, 2) * params['payoff_intensity']
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


def s_actions(_1, _2, _3, _4, signal):
    return ('actions', signal['actions'])


def s_player_0_likelihood(_1, _2, _3, _4, signal):
    return ('player_0_likelihood', signal['likelihood'][0])


def s_player_1_likelihood(_1, _2, _3, _4, signal):
    return ('player_1_likelihood', signal['likelihood'][1])

