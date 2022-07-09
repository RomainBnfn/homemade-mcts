from abc import ABCMeta, abstractmethod
import numpy as np


class AbstractBoard(metaclass=ABCMeta):

    @abstractmethod
    def get_possible_moves(self) -> np.ndarray:
        return NotImplemented(self.__class__.__name__ + '.get_possible_moves')

    @abstractmethod
    def get_next_player(self, player):
        return NotImplemented(self.__class__.__name__ + '.get_other_player')

    @abstractmethod
    def push(self, move) -> None:
        return NotImplemented(self.__class__.__name__ + '.push')

    @abstractmethod
    def pop(self, amount: int) -> None:
        return NotImplemented(self.__class__.__name__ + '.pop')

    @abstractmethod
    def get_result(self, player) -> float:
        return NotImplemented(self.__class__.__name__ + '.get_result')

    @abstractmethod
    def get_last_move(self):
        return NotImplemented(self.__class__.__name__ + '.get_last_move')

    @abstractmethod
    @property
    def is_finished(self) -> bool:
        return NotImplemented(self.__class__.__name__ + '.is_finished')
