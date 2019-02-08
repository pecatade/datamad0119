import pandas as pd

data_stats = pd.read_csv("nba-players-stats/Seasons_Stats.csv")
data_salary = pd.read_csv("nba-players-stats/NBA_season1718_salary.csv")
data_players = pd.read_csv("nba-players-stats/Players.csv")

def clean():
    data = pd.merge(data_stats, data_salary, how = "outer", on = 'Player', validate = "m:m")
    data = pd.merge(data, data_players, how = "outer", on = 'Player', validate = "m:m")
    data = data[(data['Year']== 2017)]
    data= data.drop_duplicates(subset= 'Unnamed: 0_x')
    data = data.drop(['Unnamed: 0_y','Unnamed: 0_x','FT%','Unnamed: 0','WS/48', 'OBPM', 'DBPM', 'BPM','blanl','2P%', 'eFG%','TS%','TOV%', 'USG%','blank2','Tm_y','born','ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%','BLK%','FG%','3P%'], axis=1)
    data = data.rename(index=str, columns={"seson17_18": "Salary", "Tm_x": "Team", "height": "Height", "weight": "Weight","collage": "Collage","birth_city": "Birth_City","birth_state": "Birth_State",})
    data.columns.values
    null_displ = data[(data['3PAr'].isnull()==True)]
    null_displ = data[(data['FTr'].isnull()==True)]
    data[['3PAr','FTr']] = data[['3PAr','FTr']].fillna(0.0)
    data[['Collage','Birth_City','Birth_State']] = data[['Collage','Birth_City','Birth_State']].fillna("Unknown")
    column_order = ['Year', 'Player', 'Age', 'Height','Weight','Birth_State', 'Birth_City', 'Collage', 'Team', 'Pos', 'Salary','G', 'GS', 'PTS', 'MP', 'PER', 'FTr', 'OWS', 'DWS', 'WS', 'VORP', 'FG', 'FGA', '3P', '3PA','2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB','TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
    data = data[column_order]
    data = data[data.Team != 'TOT']
    return data.to_csv('CSVS/players_stats_salary.csv', index=False)