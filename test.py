from isolation import Board
from sample_players import GreedyPlayer
from game_agent import MinimaxPlayer

p1 = MinimaxPlayer()
p2 = GreedyPlayer()
game = Board(p1, p2)

game.apply_move((0,0))
game.apply_move((0,5))

print(game.get_legal_moves())

winner,history,outcome = game.play()
print("\nWinner: {}, Outcome: {}".format(winner,outcome))
print(game.to_string())
print("History:\n{!s}".format(history))
