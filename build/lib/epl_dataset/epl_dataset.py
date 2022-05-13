import os
import numpy as np
import pandas as pd
from typing import Tuple
import torch
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)
np.random.seed(0)

def download_latest_data(data_path='./data/'):
    shell_command = f'''
    rm -rf ../tmp/Fantasy-Premier-League
    git clone https://github.com/vaastav/Fantasy-Premier-League ../tmp/Fantasy-Premier-League
    cp -r ../tmp/Fantasy-Premier-League/data/ {data_path}
    '''
    os.system(shell_command)

def normalize(x, is_scalar=False):
    """Function normalizes input to zero mean, unit variance

    Args:
        x (tensor): torch tensor
        is_scalar (bool, optional): true if scalar. Defaults to False.

    Returns:
        (normalized_x, means, stds): Returns normalized inputs and associated means and transforms
    """
    if is_scalar:
        x = x.reshape((-1, )).double() #(N, )
        means = torch.mean(x)
        stds = torch.std(x)
        normalized_x = (x - means) / stds
        return normalized_x, means, stds
    else:
        x = x.double() # (N, D, L)
        x = x.permute(0, 2, 1)
        means = torch.mean(x, dim=(0, 1))
        stds = torch.std(x, dim=(0, 1))
        normalized_x = (x - means) / (stds)
        normalized_x = normalized_x.permute(0, 2, 1)
        return normalized_x, means, stds

def get_normalized_team_names():
    """Function provides normalized team names to avoid cross year messes

    Returns:
        team_names (pd.DataFrame): dataframe containing normalized team name mapping
    """
    normalization_columns = ["id_2019","name_2019","id_2020","id_2021","normalized_team_name"]
    normalization_data = [
        ["1","Arsenal","1","1","Arsenal"],
        ["2","Aston Villa","2","2","Aston Villa"],
        ["3","Bournemouth","0","0","Bournemouth"],
        ["0","Brentford","0","3","Brentford"],
        ["4","Brighton","3","4","Brighton"],
        ["5","Burnley","4","5","Burnley"],
        ["6","Chelsea","5","6","Chelsea"],
        ["7","Crystal Palace","6","7","Crystal Palace"],
        ["8","Everton","7","8","Everton"],
        ["0","Fulham","8","0","Fulham"],
        ["9","Leicester","9","9","Leicester"],
        ["0","Leeds","10","10","Leeds"],
        ["10","Liverpool","11","11","Liverpool"],
        ["11","Man City","12","12","Manchester City"],
        ["12","Man Utd","13","13","Manchester United"],
        ["13","Newcastle","14","14","Newcastle United"],
        ["14","Norwich","0","15","Norwich"],
        ["15","Sheffield Utd","15","0","Sheffield United"],
        ["16","Southampton","16","16","Southampton"],
        ["17","Spurs","17","17","Tottenham"],
        ["18","Watford","0","18","Watford"],
        ["0","West Brom","18","0","West Bromwich Albion"],
        ["19","West Ham","19","19","West Ham"],
        ["20","Wolves","20","20","Wolverhampton Wanderers"]]
    normalized_team_names = pd.DataFrame(normalization_data, columns=normalization_columns)
    for int_column in ['id_2019','id_2020','id_2021']:
        normalized_team_names[int_column] = normalized_team_names[int_column].astype(int)
    return normalized_team_names

def get_historical_player_features(data_path, player_feature_names, max_player_points=12) -> pd.DataFrame:
    """Function reads raw data from many years, normalizes and collects processed data in one frame

    Args:
        data_path ([type]): [description]
        player_feature_names ([type]): [description]
        max_player_points (int, optional): [description]. Defaults to 12.

    Returns:
        pd.DataFrame: [description]
    """
    assert(player_feature_names[0] == "total_points")
    team_names = get_normalized_team_names()

    gameweek_data_2019 = pd.read_csv(f"{data_path}2019-20/gws/merged_gw.csv")[
        ['name', "GW", "opponent_team"] + player_feature_names]
    remove_digits = str.maketrans("", "", "0123456789")
    remove_underscore = str.maketrans("_", " ", "")
    gameweek_data_2019["GW"] = gameweek_data_2019["GW"]
    
    gameweek_data_2019['name'] =gameweek_data_2019.apply(lambda x: x['name'].translate(
        remove_underscore).translate(remove_digits).strip(), axis=1)
    gameweek_data_2019 = pd.merge(gameweek_data_2019, team_names, left_on = ['opponent_team'], right_on=['id_2019'], how="left")

    gameweek_data_2020 = pd.read_csv(f"{data_path}2020-21/gws/merged_gw.csv")[
        ['name', "GW", "opponent_team"] + player_feature_names]
    gameweek_data_2020["GW"] = gameweek_data_2020["GW"] + 53
    gameweek_data_2020 = pd.merge(gameweek_data_2020, team_names, left_on = ['opponent_team'], right_on=['id_2020'], how="left")

    game_week_data_2021 = pd.read_csv(f"{data_path}2021-22/gws/merged_gw.csv")[['name', "GW", "opponent_team"] + player_feature_names]
    game_week_data_2021['GW'] = game_week_data_2021['GW'] + (53*2)
    game_week_data_2021 = pd.merge(game_week_data_2021, team_names, left_on = ['opponent_team'], right_on=['id_2021'], how="left")

    game_weeks = pd.concat((gameweek_data_2019, gameweek_data_2020, game_week_data_2021))
    game_weeks["opponent"] = game_weeks["normalized_team_name"]
    historical_player_features = game_weeks[["name", "opponent"] + player_feature_names]
    historical_player_features.fillna(0, inplace=True)
    historical_player_features["total_points"] = historical_player_features["total_points"].clip(0, max_player_points)
    return historical_player_features

def get_dataset(data_path, input_feature_names, window_size=5, batch_size=500, num_workers=20, train_test_split=0.8, max_player_points=12, force_download=False) -> Tuple[DataLoader, DataLoader]:
    """Downloads latest english premier league dataset and generates pytorch data loaders for contextual prediction dataset 

       Each batch is of shape (batch_size, feature_size, window_size). 
       
       A good prediction task might be to use historical features as input to predict total points scored in the next game. Use total_points as first column in feature vector
        Inputs - X[:, :, :-1]
        Outputs - X[:,0,-1]

    Args:
        data_path ([type]): [description]
        input_feature_names ([type]): [description]
        window_size (int, optional): [description]. Defaults to 5.
        batch_size (int, optional): [description]. Defaults to 500.
        num_workers (int, optional): [description]. Defaults to 20.
        train_test_split (float, optional): [description]. Defaults to 0.8.
        max_player_points (int, optional): [description]. Defaults to 12.
        force_download (bool, optional): [description]. Defaults to False.

    Returns:
        Tuple[DataLoader, DataLoader]: [description]
    """
    if force_download or not os.path.exists(data_path):
        download_latest_data(data_path)
    player_features = get_historical_player_features(data_path=data_path, player_feature_names=input_feature_names, max_player_points=max_player_points)
    assert(set.issubset(set(input_feature_names), set(player_features.columns)))
    assert(input_feature_names[0] == "total_points")

    player_names = player_features['name'].unique()
    X_players = []
    for player_name in player_names:
        player_feature = player_features[player_features["name"] == player_name].transpose().values[2:] 
        for i in range(player_feature.shape[1] - window_size):
            X_players.append(player_feature[:,i:i+window_size])    
    X_players = np.array(X_players).astype(float)
    indices = np.random.permutation(range(len(X_players)))
    train_length = int(train_test_split * len(X_players))
    X_players = torch.tensor(X_players).double()
    X_players, means, stds = normalize(X_players)
    train_indices, test_indices = indices[:train_length], indices[train_length:] 
    X_train, X_test = X_players[train_indices, :, :-1], X_players[test_indices, :, :-1]
    Y_train, Y_test = X_players[train_indices, 0, -1], X_players[test_indices, 0, -1]  

    train_loader = DataLoader(TensorDataset((X_train, Y_train)), batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(TensorDataset((X_test, Y_test)), batch_size=batch_size, num_workers=num_workers)
    return train_loader, test_loader, (means, stds)

if __name__ == "__main__":
    data_path = './data2/'
    get_normalized_team_names()
    #download_latest_data(data_path)
    #train_loader, test_loader, (means, stds) = get_dataset(data_path=data_path, input_feature_names=["minutes", "ict_index", "total_points",], window_size=5, batch_size=28)
