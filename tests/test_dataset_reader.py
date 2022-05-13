import unittest
from epl_dataset import epl_dataset
import os 
import numpy as np

class TestDataProcessor(unittest.TestCase):
    def test_download_data(self):
        """Function test that download_latest_data function works
        """
        data_path='./data2/'
        epl_dataset.download_latest_data(data_path)
        self.assertEqual(os.path.exists(data_path), True)

    def test_get_player_features(self):
        """Function tests that get_player_features works

        """
        player_feature_names = ["total_points", "ict_index", 'goals_scored', 'assists', 'clean_sheets', "goals_conceded", "saves"]
        data_path='./data2/'
        max_points_per_player = np.random.randint(5, 20)
        historical_player_features = epl_dataset.get_historical_player_features(data_path, player_feature_names, max_player_points=max_points_per_player)

        # (Name, Opponent) + player_features
        self.assertEqual(historical_player_features.shape[1], len(player_feature_names) + 2)
        self.assertEqual(historical_player_features.columns[0], 'name')
        self.assertEqual(historical_player_features.columns[1], 'opponent')
        self.assertListEqual(list(historical_player_features.columns[2:]), player_feature_names)
        self.assertEqual(historical_player_features["total_points"].max(), max_points_per_player)

        # Test a burnley player
        player_name = 'Nick Pope'
        player_data = historical_player_features[historical_player_features['name'] == player_name]
        self.assertGreaterEqual(len(player_data), 50) # should have atleast more than 50 games
        self.assertGreaterEqual(player_data['saves'].sum(), 20) # decent keeper, should have atleast made 20 saves

        # Test a liverpool player
        player_name = 'Mohamed Salah'
        player_data = historical_player_features[historical_player_features['name'] == player_name]
        self.assertGreaterEqual(len(player_data), 50) # should have atleast more than 50 games
        self.assertGreaterEqual(player_data['total_points'].sum(), 200) # Salah should have scored atleast 200 points
        self.assertEqual(player_data['opponent'].values[0], 'Norwich')# In 2019 season, liverpool's first game was against norwich city FC
        self.assertLessEqual(player_data['saves'].sum(), 5) # salah should have not made many saves since he is not a goal keeper
        self.assertGreaterEqual(player_data['goals_scored'].sum(), 20) # salah should have scored more than 20 goals

    def test_get_dataset(self):
        """Function tests get data loader function
        """
        data_path = './data2/'
        epl_dataset.download_latest_data(data_path)
        for _ in range(5):
            batch_size = np.random.randint(5, 100)
            window_size = np.random.randint(3, 7)
            input_feature_names = ["total_points", "minutes", "ict_index"]
            train_loader, test_loader, (means, stds) = epl_dataset.get_dataset(data_path=data_path, input_feature_names=input_feature_names, window_size=window_size, batch_size=batch_size)
            for (x,) in train_loader:
                self.assertAlmostEqual(x.shape[0], batch_size)
                self.assertEqual(x.shape[1], len(input_feature_names))
                self.assertEqual(x.shape[2], window_size)
                break


if __name__ == '__main__':
    unittest.main()