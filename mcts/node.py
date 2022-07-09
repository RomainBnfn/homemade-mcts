from array import ArrayType
import math
from typing_extensions import Self
import numpy as np
import random as rd

from abstract_board import AbstractBoard
 
INITIAL_EXPLORATION_SCORE = 1
INITIAL_EXPLOITATION_SCORE = 0

class Node: 

    def __init__(self, move, parent_node : Self, player, absract_board : AbstractBoard, c : float) : 
        self._move = move
        self._parent_node : Node = parent_node
        self._player = player
        self._absract_board : AbstractBoard = absract_board
        self._c : float = c
        self._children : ArrayType[Node] = []
        self._score : float = 0
        self._visits : int = 0 
        self._has_been_explored : bool = False

    def get_mcts_best_child(self) -> Self:
        if(len(self._children) > 0) :
            return self._children[self.get_children_mcts_scores().argmax()]
        return self

    def get_best_known_move(self):
        if(len(self._children) > 0) :
            return self._children[self.get_children_visits().argmax()]._move
        return None

    def explore_childrens(self) -> None: 
        """Need the board to be in this state"""
        if(self._has_been_explored):
            return   
        for move in self._absract_board.get_possible_moves():
            self._children.append(
                Node(
                    move, 
                    self, 
                    self._absract_board.get_other_player(self._player), 
                    self._absract_board, 
                    self.c
                    )
                )
        self._has_been_explored = True
        return

    def simulate(self) -> None:
        """Need the board to be in the parent state"""
        self._absract_board.push(self._move)
        ##
        i = 0
        while (not self._absract_board.is_finished) :
            i += 1
            moves = self._absract_board.get_possible_moves()
            self._absract_board.push(rd.choice(moves))
        ##
        score = self._absract_board.get_result(self._player)
        self.add_score(score) 
        self.add_score_to_parents(score)
        ##
        self._absract_board.pop( i + 1 )

    def add_score(self, to_add: float) -> None:
        self._score += to_add
        self._visits += 1

    def orphan(self) -> None:
        self._parent_node = None
        
    def add_score_to_parents(self, to_add: float) -> None:
        if(self.has_parent):
            parent = self._parent_node
            while(parent.has_parent):
                parent.add_score(to_add)
                parent = self._parent_node

    def play_its_move(self) -> None : 
        self._absract_board.push(self._move)
        
    @property
    def has_parent(self) -> bool :
        return self._parent_node != None

    @property
    def is_leaf(self) -> bool :
        return self._has_been_explored and len(self._children) == 0


    @property
    def score(self) -> float:
        return self._score
        
    @property
    def visits(self) -> int:
        return self._visits
        
    @property
    def children(self) -> int:
        return self.children

    def get_exploitation_score(self) -> float:
        if(self.visits == 0):
            return INITIAL_EXPLOITATION_SCORE
        return self.score / self.visits 

    def get_exploration_score(self) -> float:
        if(self.visits == 0):
            return INITIAL_EXPLORATION_SCORE
        return self._c * math.sqrt( math.log( self._parent_node.visits ) / self.visits )
 
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