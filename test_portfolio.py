import unittest
from portfolio import Portfolio

class TestPortfolio(unittest.TestCase):
    """
    Unit tests for the Portfolio class.
    """
    def test_initialization(self):
        """
        Tests that the Portfolio class can be initialized.
        """
        portfolio = Portfolio()
        self.assertIsInstance(portfolio, Portfolio)
        self.assertEqual(portfolio.positions, {})

    def test_add_and_remove_position(self):
        """
        Tests adding and removing positions from the portfolio.
        """
        portfolio = Portfolio()
        portfolio.add_position('AAPL', 100, 150.0)
        self.assertIn('AAPL', portfolio.positions)
        self.assertEqual(portfolio.positions['AAPL']['quantity'], 100)

        portfolio.remove_position('AAPL')
        self.assertNotIn('AAPL', portfolio.positions)

    def test_update_position(self):
        """
        Tests updating a position in the portfolio.
        """
        portfolio = Portfolio()
        portfolio.add_position('GOOG', 50, 2800.0)
        portfolio.update_position('GOOG', 75, 2850.0)
        self.assertEqual(portfolio.positions['GOOG']['quantity'], 75)
        self.assertEqual(portfolio.positions['GOOG']['price'], 2850.0)

    def test_get_total_value(self):
        """
        Tests the calculation of the total portfolio value.
        """
        portfolio = Portfolio()
        portfolio.add_position('MSFT', 200, 300.0)
        portfolio.add_position('TSLA', 10, 700.0)
        expected_value = (200 * 300.0) + (10 * 700.0)
        self.assertEqual(portfolio.get_total_value(), expected_value)

if __name__ == '__main__':
    unittest.main()