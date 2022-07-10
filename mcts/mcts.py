from math import sqrt
from abstract_board import AbstractBoard
from node import Node

DEFAULT_C = sqrt(2)
AMOUNT_SIMULATION = 5
SIMULATE_ALL_CHILDREN = True


class MCTS:

    def __init__(self, board: AbstractBoard, player, c: float = DEFAULT_C) -> None:
        self._root_node: Node = Node(None, None, player, board, c)
        self._c = c
        self._board = board

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
