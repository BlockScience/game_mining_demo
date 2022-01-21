from game_mining_demo.logic import *

BLOCKS: list[dict] = [
    {
        'name': 'Play the game',
        'policies': {
            'actions': p_actions
        },
        'variables': {
            'past_actions': s_past_actions,
            'actions': s_actions,
            'players': None,
            'payoffs': s_payoffs

        }
    },
    {
        'name': 'Update beliefs & Mutate payoff tensor',
        'ignore': False,
        'policies': {
        },
        'variables': {
            'players': s_players,
            'payoff_tensor': s_payoff_tensor

        }
    }
]

BLOCKS = [block
          for block in BLOCKS
          if block.get('ignore', False) == False]
