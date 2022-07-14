from array import ArrayType
from math import sqrt

from abstract_board import AbstractBoard
from node import Node
import pickle

DEFAULT_C = sqrt(2)
AMOUNT_SIMULATION = 5
SIMULATE_ALL_CHILDREN = True
DEFAULT_SAVE_PATH = "save.pkl"


class MCTS:

    def __init__(self, board: AbstractBoard, c: float = DEFAULT_C) -> None:
        self._root_node: Node = None
        self._initial_root_node: Node = None
        self._c = c
        self._board = board
        self._define_initial_nodes()

    def _define_initial_nodes(self):
        initial_moves = self._board.get_previous_moves()
        node = Node(None, None, None, self._board, self._c)
        self._initial_root_node: Node = node
        self._board.reset()
        for move in initial_moves:
            player = self._board.get_actual_player()
            node = Node(move, node, player, self._board, self._c)
        self._root_node = node

    def save(self, path: str = DEFAULT_SAVE_PATH) -> None:
        """
        Save the actual mcts tree into a binary filed (to be loaded after)
        :param path: The path of the saved file
        """
        with open(path, "wb") as file:
            pickle.dump(self._initial_root_node, file)
        print("MCTS tree saved !")

    def load(self, path: str = DEFAULT_SAVE_PATH) -> None:
        """
        Load a mcts tree. The first player need to be the same to me loaded
        :param path: Path of saved file
        """
        with open(path, "rb") as file:
            saved_root_node: Node = pickle.load(file)
        moves = self._board.get_previous_moves()
        can_continue = True
        node = saved_root_node

        # Saving actual moves in case if we can't load the saved tree
        actual_moves = self._board.get_previous_moves()
        self._board.reset()

        while len(moves) > 0 and can_continue:
            move = moves.pop()
            self._board.play_move(move)
            children: ArrayType[Node] = node.children
            found_node: Node | None = None
            while len(children) > 0 and found_node is None:
                child: Node = children.pop()
                if child.move == move:
                    found_node = child
            can_continue = found_node is None
            if can_continue:
                node = found_node
        if not can_continue and node == saved_root_node:
            # No one of the first saved node children correspond to our first move
            # ==> Not the same game
            self._board.reset()
            self._board.play_moves(actual_moves)
            print('Unable to load the saved tree, maybe the first player is not the same.')
        while len(moves) > 0:
            # The beginning of the game is the same as the saved tree but
            # at a moment, following nodes are unknown
            move = moves.pop()
            self._board.play_move(move)
            player = self._board.get_actual_player()
            node = Node(move, node, player, self._board, self._c)
        self._initial_root_node = saved_root_node
        self._root_node = node
        print('Tree loaded without issue.')

    def learn(self, amount_learns: int = 1, amount_simulations: int = AMOUNT_SIMULATION) -> None:
        """
        Use MCTS algorithm to learn the best next moves
        :param amount_learns:       Amount of applications of MCTS algorithm
        :param amount_simulations:  Amount of roll-out simulations
        """
        [self._learn_once(amount_simulations) for _ in range(amount_learns)]

    def _learn_once(self, amount_simulations: int) -> None:
        """
        Application of MCTS algorithm once
        :param amount_simulations: Amount of roll-out simulations
        """
        # Selection part
        node = self._root_node
        while node.has_been_explored or not node.is_leaf:
            node = node.get_best_mcts_child()

        # Expansion part
        node.explore_children()

        # Simulation part
        score = 0
        amount_simulated_children = 1
        if SIMULATE_ALL_CHILDREN:
            amount_simulated_children = len(node.children)
            for child in node.children:
                score += child.simulate(amount_simulations)
        else:
            score = node.get_best_mcts_child().simulate(amount_simulations)

        # Back propagation part
        node.add_score_to_parents(score, amount_simulations * amount_simulated_children)

    def get_best_known_move(self):
        """
        :return: The move of the most visited of next nodes
        """
        return self._root_node.get_best_known_move()

    def redefine_root_node(self, opponent_move) -> None:
        """
        Redefine the root node of MCST after that a player play a move
        :param opponent_move:
        :return:
        """
        if self._root_node.is_leaf:
            raise Exception('Trying to redefine a leaf & root node (game is over)')
        if not self._root_node.has_been_explored:
            other_player = self._board.get_next_player(self._root_node.player)
            self._root_node = Node(opponent_move, self._root_node, other_player, self._board, self._c)
            return
        for node in self._root_node.children:
            if node.move == opponent_move:
                self._root_node = node
                return
        raise Exception('Opponent played an unknown move while children has been visited')

