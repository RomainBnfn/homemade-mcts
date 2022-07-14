from abc import ABCMeta, abstractmethod
from array import ArrayType

import numpy as np


class AbstractBoard(metaclass=ABCMeta):

    @abstractmethod
    def get_possible_moves(self) -> np.ndarray:
        return NotImplemented(self.__class__.__name__ + '.get_possible_moves')

    @abstractmethod
    def get_actual_player(self):
        return NotImplemented(self.__class__.__name__ + '.get_actual_player')

    @abstractmethod
    def get_next_player(self, player):
        return NotImplemented(self.__class__.__name__ + '.get_other_player')

    @abstractmethod
    def play_move(self, move) -> None:
        return NotImplemented(self.__class__.__name__ + '.push')

    @abstractmethod
    def reset(self) -> None:
        return NotImplemented(self.__class__.__name__ + '.pop')

    @abstractmethod
    def get_score(self, player) -> float:
        """Score needs to be inside [0,1]"""
        return NotImplemented(self.__class__.__name__ + '.get_result')

    @abstractmethod
    @property
    def is_finished(self) -> bool:
        return NotImplemented(self.__class__.__name__ + '.is_finished')

    @abstractmethod
    def get_previous_moves(self) -> ArrayType:
        return NotImplemented(self.__class__.__name__ + '.get_previous_moves')

    def play_moves(self, moves: ArrayType):
        for move in moves:
            self.play_move(move)
