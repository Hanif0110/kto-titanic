import unittest
from exercices.mon_premier_script_avec_fonction import count_names_longer_than


class TestNames(unittest.TestCase):
    def test_count_names_longer_than(self):
        prenoms = [
            "Guillaume",
            "Gilles",
            "Juliette",
            "Antoine",
            "François",
            "Cassandre",
        ]
        result = count_names_longer_than(prenoms)
        self.assertEqual(result, 4)


if __name__ == "__main__":
    unittest.main()
