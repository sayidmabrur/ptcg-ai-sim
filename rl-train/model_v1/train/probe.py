# from engine.cg.game import battle_start
# from pprint import pprint
# from helpers.helper import build_deck
# # from model_v1.
#
#
# player1_deck = build_deck("./decks/main_agent.csv")
# player2_deck = build_deck("./decks/opponents/alakazam.csv")
#
#
# step = 0
# obs, data = battle_start(player1_deck, player2_deck)
#
#
# while obs['current'] is None or obs['current']['result'] == -1:
#
#
# pprint(data)
# pprint("="*50)
# pprint(obs)#


import random
from pprint import pprint
from engine.cg.game import battle_start, battle_select, battle_finish
from helpers.helper import build_deck

obs, start = battle_start(
    build_deck("./decks/main_agent.csv"), build_deck("./decks/opponents/alakazam.csv")
)
step = 0
while obs["current"] is None or obs["current"]["result"] == -1:
    sel = obs["select"]
    if step == 5:
        pprint(obs)
        pprint('='*30)
        print("selection:", sel)
    n = random.randint(sel["minCount"], sel["maxCount"])
    choice = random.sample(range(len(sel["option"])), n)
    obs = battle_select(choice)
    # pprint(
    #     {
    #         "step": step,
    #         "player": obs["current"]["yourIndex"] if obs["current"] else None,
    #         "logs": obs["logs"],
    #     }
    # )
    pprint("="*30)
    if step == 5:
        break
    step += 1
print("winner:", obs["current"]["result"])
battle_finish()
