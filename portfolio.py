import numpy as np

class Portfolio:
    """
    Manages a collection of financial assets, allowing for the addition, removal,
    and valuation of positions.

    The Portfolio class provides functionalities to track the quantity and price
    of various assets, calculate the total portfolio value, and create deep
    copies for scenario analysis.
    """
    def __init__(self):
        """
        Initializes an empty portfolio with no positions.
        """
        self.positions = {}

    def clone(self):
        """
        Creates a deep copy of the portfolio.

        This method is essential for non-destructive operations, such as stress
        testing, where modifications to a temporary portfolio should not affect
        the original.

        Returns:
            Portfolio: A new Portfolio instance with the same positions.
        """
        new_portfolio = Portfolio()
        new_portfolio.positions = {k: v.copy() for k, v in self.positions.items()}
        return new_portfolio

    def add_position(self, symbol, quantity, price):
        """
        Adds a new position or updates an existing one in the portfolio.

        If the symbol already exists, it delegates to the `update_position`
        method to avoid duplicating logic.

        Args:
            symbol (str): The stock symbol (e.g., 'AAPL').
            quantity (int): The number of shares.
            price (float): The current price per share.
        """
        if symbol in self.positions:
            self.update_position(symbol, quantity, price)
        else:
            self.positions[symbol] = {'quantity': quantity, 'price': price}

    def remove_position(self, symbol):
        """
        Removes a position from the portfolio by its symbol.

        If the symbol is not found, the method does nothing.

        Args:
            symbol (str): The stock symbol to remove.
        """
        if symbol in self.positions:
            del self.positions[symbol]

    def update_position(self, symbol, quantity, price):
        """
        Updates the quantity and price of an existing position.

        If the symbol does not exist, the method does nothing.

        Args:
            symbol (str): The stock symbol to update.
            quantity (int): The new quantity of shares.
            price (float): The new price per share.
        """
        if symbol in self.positions:
            self.positions[symbol]['quantity'] = quantity
            self.positions[symbol]['price'] = price

    def get_total_value(self):
        """
        Calculates the total market value of all positions in the portfolio.

        The value is computed by summing the product of quantity and price for
        each position.

        Returns:
            float: The total portfolio value.
        """
        total_value = 0
        for symbol, data in self.positions.items():
            total_value += data['quantity'] * data['price']
        return total_value