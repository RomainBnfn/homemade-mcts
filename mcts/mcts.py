from math import sqrt
from abstract_board import AbstractBoard
from node import Node

DEFAULT_C = sqrt(2)


class MCTS:

    def __init__(self, board: AbstractBoard, player, c: float = DEFAULT_C) -> None:
        self._root_node: Node = Node(None, None, player, board, c)
        self._board = board

    def learn(self, amount: int = 1) -> None:
        [self._learn_once() for _ in range(amount)]

    def _learn_once(self) -> None:
        # Selection part
        node = self._root_node
        i = 0
        while node.has_been_explored or not node.is_leaf:
            node = node.get_mcts_best_child()
            node.play_its_move()
            i += 1

        # Expansion part
        node.explore_children()

        # Simulation part
        node.get_mcts_best_child().simulate()

        # Back propagation part is include in node simulation
        self._board.pop(i)

    def get_best_known_move(self):
        return self._root_node.get_best_known_move()

    def redefine_root_node(self, opponent_move) -> None:
        if self._root_node.is_leaf:
            raise Exception('Trying to redefine a leaf & root node')
        if not self._root_node.has_been_explored:
            other_player = self._board.get_next_player(self._root_node.player)
            self._root_node = Node(opponent_move, None, other_player, self._board, self.c)
            return
        for node in self._root_node.children:
            if node.move == opponent_move:
                node.orphan()
                self._root_node = node
                return
        raise Exception('Opponent played an unknown move')
