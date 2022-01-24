from game_mining_demo.types import PlayerParams, Player

from cadCAD_tools.preparation import sweep_cartesian_product

TIMESTEPS = 500
SAMPLES = 5

PLAYER_1_PAYOFF = [[1, 0.5], [0.5, 0.1]]
PLAYER_2_PAYOFF = [[1, 0.5], [0.5, 0.1]]

PAYOFF_TENSOR = [PLAYER_1_PAYOFF, PLAYER_2_PAYOFF]
# dim: [player_no, player_1_action, player_2_action]


# %%

N_PLAYERS = 2
INITIAL_LIKELIHOOD = 0.5

INITIAL_LIKELIHOOD_VECTOR = {i: INITIAL_LIKELIHOOD
                             for i in range(N_PLAYERS)}

DEFAULT_PLAYER_PARAMS = PlayerParams(2, 4, 4, 10, 5)

INITIAL_PLAYERS = [
    Player(i, INITIAL_LIKELIHOOD_VECTOR, DEFAULT_PLAYER_PARAMS)
    for i in range(N_PLAYERS)
]

PARAMS: dict = {
    'payoff_shift_interval': [10, 100],
    'payoff_intensity': [0.1, 2.0]
}

PARAMS = sweep_cartesian_product(PARAMS)


INITIAL_STATE: dict = {
    'players': INITIAL_PLAYERS,
    'actions': {},
    'past_actions': {0: [], 1: []},
    'payoffs': {},
    'payoff_tensor': PAYOFF_TENSOR
}

