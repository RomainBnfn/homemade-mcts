import math
from typing import Any
from typing_extensions import Self
import numpy as np

INITIAL_EXPLORATION_SCORE = 1
INITIAL_EXPLOITATION_SCORE = 1


class Node:

    def __init__(self, move, parent_node: Self, player):
        self._move = move
        self._parent_node: Node = parent_node
        self._player = player
        self._children: list[Node] = []
        self._score: float = 0
        self._visits: int = 0
        self._has_been_explored: bool = False

    def get_best_mcts_child(self, c: float) -> Self:
        """:return: The child node which has the best exploration + exploitation score
        or itself is it's a leaf node."""
        if len(self._children) > 0:
            return self._children[self.get_children_mcts_scores(c).argmax()]
        return self

    def get_best_known_move(self):
        """:return: The child node which has the higher amount of visits, or None is
        it's a leaf node."""
        if len(self._children) > 0:
            return self._children[self.get_children_visits().argmax()].move
        return None

    def get_all_moves(self) -> list[Any]:
        """Get all moves from initial_node to this node."""
        moves = []
        node = self
        while node.has_parent:
            node = node._parent_node
            if node.move is not None:
                moves.insert(0, node.move)
        return moves

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
    def children(self) -> list[Self]:
        return self._children

    @children.setter
    def children(self, value: list):
        self._children = value

    @property
    def move(self):
        return self._move

    @property
    def has_been_explored(self):
        return self._has_been_explored

    @has_been_explored.setter
    def has_been_explored(self, value: bool):
        self._has_been_explored = value

    @property
    def player(self):
        return self._player

    def get_exploitation_score(self) -> float:
        if self.visits == 0:
            return INITIAL_EXPLOITATION_SCORE
        return self.score / self.visits

    def get_exploration_score(self, c: float) -> float:
        if self.visits == 0:
            return INITIAL_EXPLORATION_SCORE
        return c * math.sqrt(math.log(self._parent_node.visits) / self.visits)

    def get_mcts_score(self, c: float) -> float:
        return self.get_exploitation_score() + self.get_exploration_score(c)

    def get_children_mcts_scores(self, c: float) -> np.ndarray:
        scores = []
        for child in self._children:
            scores.append(child.get_mcts_score(c))
        return np.array(scores)

    def get_children_visits(self) -> np.ndarray:
        visits = []
        for child in self._children:
            visits.append(child.visits)
        return np.array(visits)
