from array import ArrayType
import math
from typing_extensions import Self
import numpy as np
import random as rd

from abstract_board import AbstractBoard

INITIAL_EXPLORATION_SCORE = 1
INITIAL_EXPLOITATION_SCORE = 1


class Node:

    def __init__(self, move, parent_node: Self, player, board: AbstractBoard, c: float):
        self._move = move
        self._parent_node: Node = parent_node
        self._player = player
        self._board: AbstractBoard = board
        self._c: float = c
        self._children: ArrayType[Node] = []
        self._score: float = 0
        self._visits: int = 0
        self._has_been_explored: bool = False

    def get_best_mcts_child(self) -> Self:
        """:return: The child node which has the best exploration + exploitation score
        or itself is it's a leaf node."""
        if len(self._children) > 0:
            return self._children[self.get_children_mcts_scores().argmax()]
        return self

    def get_best_known_move(self):
        """:return: The child node which has the higher amount of visits, or None is
        it's a leaf node."""
        if len(self._children) > 0:
            return self._children[self.get_children_visits().argmax()].move
        return None

    def explore_children(self) -> None:
        """Create children next nodes according to the board possibles moves, and
        reset the board after-while."""
        if self._has_been_explored:
            return
        self.go_into_its_state()
        for move in self._board.get_possible_moves():
            self._children.append(
                Node(
                    move,
                    self,
                    self._board.get_next_player(self._player),
                    self._board,
                    self._c
                )
            )
        self._board.reset()
        self._has_been_explored = True
        return

    def go_into_its_state(self) -> None:
        """Play all parents moves to reach its state."""
        moves = []
        node = self
        while node.has_parent:
            node = node._parent_node
            moves.insert(0, node.move)
        for move in moves:
            self._board.play_move(move)

    def simulate(self, max_depth: int = -1, amount_simulations: int = 1) -> float:
        total_score = 0
        for _ in range(amount_simulations):
            self.go_into_its_state()
            remaining_depth = max_depth
            while not self._board.is_finished or remaining_depth != 0:
                moves = self._board.get_possible_moves()
                self._board.play_move(rd.choice(moves))
                remaining_depth -= 1
            score = self._board.get_score(self._player)
            total_score += score
            self.add_score(score)
            self._board.reset()
        return total_score

    def add_score(self, score: float, amount_simulations: int = 1) -> None:
        self._score += score
        self._visits += amount_simulations

    def orphan(self) -> None:
        self._parent_node = None

    def add_score_to_parents(self, score: float, amount_simulations: int = 1) -> None:
        if self.has_parent:
            parent = self._parent_node
            while parent.has_parent:
                parent.add_score(score, amount_simulations)
                parent = self._parent_node

    def play_its_move(self) -> None:
        self._board.play_move(self._move)

    @property
    def has_parent(self) -> bool:
        return self._parent_node is not None

    @property
    def is_leaf(self) -> bool:
        return self._has_been_explored and len(self._children) == 0

    @property
    def score(self) -> float:
        return self._score

    @property
    def visits(self) -> int:
        return self._visits

    @property
    def children(self) -> ArrayType[Self]:
        return self._children

    @property
    def move(self):
        return self._move

    @property
    def has_been_explored(self):
        return self._has_been_explored

    @property
    def player(self):
        return self._player

    def get_exploitation_score(self) -> float:
        if self.visits == 0:
            return INITIAL_EXPLOITATION_SCORE
        return self.score / self.visits

    def get_exploration_score(self) -> float:
        if self.visits == 0:
            return INITIAL_EXPLORATION_SCORE
        return self._c * math.sqrt(math.log(self._parent_node.visits) / self.visits)

    def get_mcts_score(self) -> float:
        return self.get_exploitation_score() + self.get_exploration_score()

    def get_children_mcts_scores(self) -> np.ndarray:
        scores = []
        for child in self._children:
            scores.append(child.get_mcts_score())
        return np.array(scores)

    def get_children_visits(self) -> np.ndarray:
        visits = []
        for child in self._children:
            visits.append(child.visits)
        return np.array(visits)
