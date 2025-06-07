from abc import ABC, abstractmethod

class BaseOptionModel(ABC):
    """
    Abstract base class for all option pricing models.
    """
    @abstractmethod
    def price(self):
        """
        Calculate the option price.
        """
        pass

    @abstractmethod
    def calculate_greeks(self):
        """
        Calculate the option Greeks.
        """
        pass