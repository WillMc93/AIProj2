"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
	"""Subclass base exception for code clarity. """
	pass


def custom_score(game, player):
	"""This heuristic just maximizes the number of moves for current player.

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

	# Return number of moves for current player
	return float(len(game.get_legal_moves(player)))


def custom_score_2(game, player):
	"""This heuristic maximizes current players moves over opponents number of moves.

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
		return float("-inf")

	if game.is_winner(player):
		return float("inf")

	own_moves = len(game.get_legal_moves(player))
	opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

	return float(own_moves - opp_moves)


def custom_score_3(game, player):
	"""Calculate the heuristic value of a game state from the point of view
	of the given player.

	Note: this function should be called from within a Player instance as
	`self.score()` -- you should not need to call this function directly.

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
	raise NotImplementedError


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
	def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
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

		**************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

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
		best_move = (-1, -1)

		try:
			# The try/except block will automatically catch the exception
			# raised when the timer is about to expire.
			best_move = self.minimax(game, self.search_depth)
			return best_move

		except SearchTimeout:
			pass  # Handle any actions required after timeout as needed

		# Return the best move from the last completed search iteration
		return best_move

	def search_layer(self, game, depth, maximize=True):
		"""Implementation of the depth-limited minimax search algorithm as described in
		the lectures. This function recursively adds search layers serving
		as either min_value or max_value as described in the AIMA text depending on the
		value of maximize.

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
		float
			The score of the best move found.
		"""

		if self.time_left() < self.TIMER_THRESHOLD:
			raise SearchTimeout()
			#return self.score(game, self)

		# Return the score for this game if there are no legal moves
		if not game.get_legal_moves():
			return self.score(game, self)

		# If at end of search tree, return the score for this move
		if depth == 0:
			return self.score(game, self)

		# Initialize score (v in AIMA text) variable to +/- infinity depending on the value of maximize
		score = float('-inf') if maximize else float('inf')
		# Set the optimization function depending on the value of maximize
		optimize = max if maximize else min

		# Walk the tree
		for move in game.get_legal_moves():
			# To walk the tree we must decrement depth and inverse maximize.
			score = optimize(score, self.search_layer(game.forecast_move(move), depth-1, maximize=not maximize))

		# Return the best score
		return score

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

		# To keep track of the moves, declare the moves dictionary.
		# This dictionary will use the score obtained from self.search_layer()
		# as the key and the move as the value so that we may sort by score.
		moves_dict = dict()

		for move in game.get_legal_moves():
			# Run search. Param maximize must be False because next layer will always
			# be minimizing from this level
			score = self.search_layer(game.forecast_move(move), depth-1, maximize=False)

			# Add score as key and move as value to the moves_dict
			moves_dict[score] = move

		# Sort moves_list's keys (putting the max score at the end of the list)
		# and get the max
		max_score = sorted(moves_dict.keys())[-1]

		# Return the move with the max score.
		return moves_dict[max_score]


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
		# TODO: finish this function!
		# I mean . . . alright.
		# Ctrl+C, Ctrl+V, type-type-type. Yep, that looks good.
		# Better add a snarky comment for good measure

		self.time_left = time_left

		# Initialize the best move so that this function returns something
		# in case the search fails due to timeout
		best_move = (-1, -1)

		try:
			# The try/except block will automatically catch the exception
			# raised when the timer is about to expire.
			#best_move = self.iterative_deepening(self.alphabeta, game)
			best_move = self.alphabeta(game, 3)

		except SearchTimeout:
			pass  # Handle any actions required after timeout as needed

		# Return the best move from the last completed search iteration
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
			return (-1,-1)

		# To keep track of the moves, declare the moves dictionary.
		# This dictionary will use the score obtained from self.search_layer()
		# as the key and the move as the value so that we may sort by score.
		moves_dict = dict()

		for move in game.get_legal_moves():
			# Run search. Param maximize must be True because next layer will always
			# be maximizing from this level
			# NOTE: this is the opposite setup of Minimax
			score = self.search_layer(game.forecast_move(move), depth-1, maximize=True)

			# Add score as key and move as value to the moves_dict
			moves_dict[score] = move

		# Sort moves_list's keys (putting the max score at the end of the list)
		# and get the max
		max_score = sorted(moves_dict.keys())[-1]

		# Return the move with the max score.
		return moves_dict[max_score]

	def search_layer(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximize=True):
		"""Implementation of the depth-limited alphabeta search algorithm as described in
		the lectures. This function recursively adds search layers serving
		as either min_value or max_value as described in the AIMA text depending on the
		value of maximize.

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
		float
			The score of the best move found.
		"""

		if self.time_left() < self.TIMER_THRESHOLD:
			raise SearchTimeout()

		# Return the score for this game if there are no legal moves
		if not game.get_legal_moves():
			return self.score(game, self)

		# If at end of search tree, return the score for this move
		if depth == 0:
			return self.score(game, self)

		# Initialize score (v in AIMA text) variable to +/- infinity depending on the value of maximize
		score = float('-inf') if maximize else float('inf')
		# Set the optimization function depending on the value of maximize
		optimize = max if maximize else min

		# Walk the tree
		for move in game.get_legal_moves():
			# To walk the tree we must decrement depth and inverse maximize.
			score = optimize(score, self.search_layer(game.forecast_move(move), depth-1, alpha, beta, maximize=not maximize))

			# Check score against beta/alpha as applicable depending on value of maximize
			if maximize:
				if score >= beta:
					return score
				alpha = max(alpha, score)
			else:
				if score <= alpha:
					return score
				beta = min(beta, score)

		# Return the best score
		return score

	def iterative_deepening(self, search_function, game):
		"""
		This function also implements the ITERATIVE-DEEPENING-SEARCH in the AIMA text
		https://github.com/aimacode/aima-psuedocode/blob/master/md/Iterative-Deepening-Search.md

		"""
		# Not sure if really necessary, but directions said all helper functions and is helper function. So . . .
		if self.time_left() < self.TIMER_THRESHOLD:
			raise SearchTimeout()

		# Initialize depth to start very shallow
		depth = 1

		# Initialize best_move to the 'null' move
		best_move = (-1, -1)
		try:
			while True:
				best_move = search_function(game, depth)
				depth += 1
		except SearchTimeout:
			pass

		return best_move
