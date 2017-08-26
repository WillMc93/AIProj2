import random


class SearchTimeout(Exception):
	"""Subclass base exception for code clarity. """
	pass


def custom_score(game, player):
	"""This heuristic just maximizes the number of moves for current player over opponents.

	Parameters
	----------
	game : `isolation.Board`
		An instance of `isolation.Board` encoding the current state of the
		game (e.g., player locations and blocked cells).

	player : object
		A player instance in the current game (i.e., an object corresponding to
		one of the player objects `game.__player_1__` or `game.__player_2__`.)

	Returns
	-------
	float
		The heuristic value of the current game state to the specified player.
	"""
	# TODO: finish this function!
	if game.is_loser(player):
		return float('-inf')

	if game.is_winner(player):
		return float('inf')

	# Get each players moves
	own_moves = len(game.get_legal_moves(player))
	opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

	# Return difference in number of moves
	return float(own_moves - opp_moves)


def custom_score_2(game, player):
	"""This heuristic rewards moves that increase distance from the corner and decrease the opponents distance to the corners.


	Parameters
	----------
	game : `isolation.Board`
		An instance of `isolation.Board` encoding the current state of the
		game (e.g., player locations and blocked cells).

	player : object
		A player instance in the current game (i.e., an object corresponding to
		one of the player objects `game.__player_1__` or `game.__player_2__`.)

	Returns
	-------
	float
		The heuristic value of the current game state to the specified player.
	"""
	if game.is_loser(player):
		return float('-inf')

	if game.is_winner(player):
		return float('inf')

	# Define the corners
	corners = [(0, 0), (0, game.width - 1), (game.height - 1, 0), (game.height - 1, game.width - 1)]

	# Get each players moves
	own_moves = game.get_legal_moves(player)
	opp_moves = game.get_legal_moves(game.get_opponent(player))

	# Get number of moves for each player that are in the corner
	own_corners = len([move for move in own_moves if move in corners])
	opp_corners = len([move for move in opp_moves if move in corners])

	# Get the fraction of spaces available
	# This allows us to prioritize central moves as the game progresses
	game_state = game.width * game.height / len(game.get_blank_spaces())

	# Return the difference in scores + the scaled difference in moves that result in corners.
	return float(len(own_moves) - len(opp_moves) + game_state * (opp_corners - own_corners))


def custom_score_3(game, player):
	"""This heuristic increases the our own euclidean distance from the corners, and
	increases our opponent's.

	Parameters
	----------
	game : `isolation.Board`
		An instance of `isolation.Board` encoding the current state of the
		game (e.g., player locations and blocked cells).

	player : object
		A player instance in the current game (i.e., an object corresponding to
		one of the player objects `game.__player_1__` or `game.__player_2__`.)

	Returns
	-------
	float
		The heuristic value of the current game state to the specified player.
	"""
	if game.is_loser(player):
		return float('-inf')

	if game.is_winner(player):
		return float('inf')

	# Define the corners
	corners = [(0, 0), (0, game.width - 1), (game.height - 1, 0), (game.height - 1, game.width - 1)]

	own_moves = game.get_legal_moves(player)
	opp_moves = game.get_legal_moves(game.get_opponent(player))

	own_score = 0
	opp_score = 0

	def get_euclid(moves):
		if len(moves) == 0:
			return 0

		move_score_sum = 0
		for move in moves:
			move_score = 0
			for corner in corners:
				# Calculate the euclidean distnace from the corner
				distance = abs(move[0] - corner[0]) ** 2 + abs(move[1] - corner[1]) ** 2
				distance = distance ** 0.5

				# Sum the distance
				move_score += distance

			# Get sum of the average move_scores
			move_score_sum += move_score / 4

		# Return average distance
		return float(move_score_sum / len(moves))



	own_score = get_euclid(own_moves)
	opp_score = get_euclid(opp_moves)

	return float(own_score - opp_score)

class IsolationPlayer:
	"""Base class for minimax and alphabeta agents -- this class is never
	constructed or tested directly.

	********************  DO NOT MODIFY THIS CLASS  ********************

	Parameters
	----------
	search_depth : int (optional)
		A strictly positive integer (i.e., 1, 2, 3,...) for the number of
		layers in the game tree to explore for fixed-depth search. (i.e., a
		depth of one (1) would only explore the immediate sucessors of the
		current state.)

	score_fn : callable (optional)
		A function to use for heuristic evaluation of game states.

	timeout : float (optional)
		Time remaining (in milliseconds) when search is aborted. Should be a
		positive value large enough to allow the function to return before the
		timer expires.
	"""
	def __init__(self, search_depth=3, score_fn=custom_score_2, timeout=100.):
		self.search_depth = search_depth
		self.score = score_fn
		self.time_left = None
		self.TIMER_THRESHOLD = timeout

class MinimaxPlayer(IsolationPlayer):
	"""Game-playing agent that chooses a move using depth-limited minimax
	search. You must finish and test this player to make sure it properly uses
	minimax to return a good move before the search time limit expires.
	"""

	def get_move(self, game, time_left):
		"""Search for the best move from the available legal moves and return a
		result before the time limit expires.

		For fixed-depth search, this function simply wraps the call to the
		minimax method, but this method provides a common interface for all
		Isolation agents, and you will replace it in the AlphaBetaPlayer with
		iterative deepening search.

		Parameters
		----------
		game : `isolation.Board`
			An instance of `isolation.Board` encoding the current state of the
			game (e.g., player locations and blocked cells).

		time_left : callable
			A function that returns the number of milliseconds left in the
			current turn. Returning with any less than 0 ms remaining forfeits
			the game.

		Returns
		-------
		(int, int)
			Board coordinates corresponding to a legal move; may return
			(-1, -1) if there are no available legal moves.
		"""
		self.time_left = time_left

		# Initialize the best move so that this function returns something
		# in case the search fails due to timeout
		if not game.get_legal_moves():
			return (-1, -1)
		best_move = game.get_legal_moves()[0]

		try:
			# Run search
			best_move = self.minimax(game, self.search_depth)
			return best_move

		except SearchTimeout:
			return best_move

	def minimax(self, game, depth):
		"""This function begins the search for the best move.


		This should be a modified version of MINIMAX-DECISION in the AIMA text.
		https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

		Parameters
		----------
		game : isolation.Board
			An instance of the Isolation game `Board` class representing the
			current game state

		depth : int
			Depth is an integer representing the maximum number of plies to
			search in the game tree before aborting

		maximize : bool
			Set this search layer to maximize (True) or minimize (False).

		Returns
		-------
		(int, int)
			The board coordinates of the best move found in the current search;
			(-1, -1) if there are no legal moves

		Notes
		-----
			(1) You MUST use the `self.score()` method for board evaluation
				to pass the project tests; you cannot call any other evaluation
				function directly.

			(2) If you use any helper functions (e.g., as shown in the AIMA
				pseudocode) then you must copy the timer check into the top of
				each helper function or else your agent will timeout during
				testing.
		"""
		if self.time_left() < self.TIMER_THRESHOLD:
			raise SearchTimeout()

		if not game.get_legal_moves():
			return (-1, -1)

		def max_value(game, depth):
			if self.time_left() < self.TIMER_THRESHOLD:
				raise SearchTimeout()

			if not game.get_legal_moves() or depth == 0:
				return self.score(game, self)

			score = float('-inf')
			for move in game.get_legal_moves():
				score = max(score, min_value(game.forecast_move(move), depth-1))

			return score

		def min_value(game, depth, alpha=float('-inf'), beta=float('inf')):
			if self.time_left() < self.TIMER_THRESHOLD:
				raise SearchTimeout()

			if not game.get_legal_moves() or depth == 0:
				return self.score(game, self)

			score = float('inf')
			for move in game.get_legal_moves():
				score = min(score, max_value(game.forecast_move(move), depth-1))

			return score

		# To keep track of the best score, move combo declare a tuple of (score, move)
		besties = (float('-inf'), game.get_legal_moves()[0])

		for move in game.get_legal_moves():
			# Run search.
			score = min_value(game.forecast_move(move), depth-1)

			if score > besties[0]:
				besties = (score, move)

		return besties[1]


class AlphaBetaPlayer(IsolationPlayer):
	"""Game-playing agent that chooses a move using iterative deepening minimax
	search with alpha-beta pruning. You must finish and test this player to
	make sure it returns a good move before the search time limit expires.
	"""

	def get_move(self, game, time_left, maximize=True):
		"""Search for the best move from the available legal moves and return a
		result before the time limit expires.

		Modify the get_move() method from the MinimaxPlayer class to implement
		iterative deepening search instead of fixed-depth search.

		**********************************************************************
		NOTE: If time_left() < 0 when this function returns, the agent will
			  forfeit the game due to timeout. You must return _before_ the
			  timer reaches 0.
		**********************************************************************

		Parameters
		----------
		game : `isolation.Board`
			An instance of `isolation.Board` encoding the current state of the
			game (e.g., player locations and blocked cells).

		time_left : callable
			A function that returns the number of milliseconds left in the
			current turn. Returning with any less than 0 ms remaining forfeits
			the game.

		Returns
		-------
		(int, int)
			Board coordinates corresponding to a legal move; may return
			(-1, -1) if there are no available legal moves.
		"""
		self.time_left = time_left

		# Initialize the best move so that this function returns something
		# in case the search fails due to timeout
		if not game.get_legal_moves():
			return (-1, -1)
		best_move = game.get_legal_moves()[0]

		depth = 1
		try:
			while True:
				best_move = self.alphabeta(game, depth)
				depth += 1

		except SearchTimeout:
			# Return the best move from the last completed search
			return best_move


	def alphabeta(self, game, depth):
		"""Implement iterative-deeping, depth-limited minimax search with alpha-beta pruning as
		described in the lectures.

		This is a modified version of ALPHA-BETA-SEARCH in the AIMA text
		https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

		Parameters
		----------
		game : isolation.Board
			An instance of the Isolation game `Board` class representing the
			current game state

		depth : int
			Depth is an integer representing the maximum number of plies to
			search in the game tree before aborting

		alpha : float
			Alpha limits the lower bound of search on minimizing layers

		beta : float
			Beta limits the upper bound of search on maximizing layers

		maximize : bool
			Sets this search layer to maxmize (True) or minimize (False).

		Returns
		-------
		(int, int)
			The board coordinates of the best move found in the current search;
			(-1, -1) if there are no legal moves
		"""
		if self.time_left() < self.TIMER_THRESHOLD:
			raise SearchTimeout()

		if not game.get_legal_moves():
			return (-1, -1)

		def max_value(game, depth, alpha=float('-inf'), beta=float('inf')):
			if self.time_left() < self.TIMER_THRESHOLD:
				raise SearchTimeout()

			if not game.get_legal_moves() or depth == 0:
				return self.score(game, self)

			score = float('-inf')
			for move in game.get_legal_moves():
				score = max(score, min_value(game.forecast_move(move), depth-1, alpha=alpha, beta=beta))

				if score >= beta:
					return score

				alpha = max(alpha, score)

			return score

		def min_value(game, depth, alpha=float('-inf'), beta=float('inf')):
			if self.time_left() < self.TIMER_THRESHOLD:
				raise SearchTimeout()

			if not game.get_legal_moves() or depth == 0:
				return self.score(game, self)

			score = float('inf')
			for move in game.get_legal_moves():
				score = min(score, max_value(game.forecast_move(move), depth-1, alpha=alpha, beta=beta))

				if score <= alpha:
					return score

				beta = min(beta, score)

			return score

		# To keep track of the best score, move combo declare a tuple of (score, move)
		besties = (float('-inf'), (-1, -1))
		# Def alpha for first max layer
		alpha = float('-inf')

		for move in game.get_legal_moves():
			# Run search.
			score = min_value(game.forecast_move(move), depth-1, alpha=alpha)

			if score >= besties[0]:
				besties = (score, move)

			alpha = max(alpha, score)

		# Return the move with the max score.
		return besties[1]
