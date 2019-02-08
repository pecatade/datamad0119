import pandas as pd
import numpy as np

data = pd.read_csv("CSVS/players_stats_salary.csv")

def team(team):
    t = data[(data["Team"] == team)]
    datafloat = t[['Salary','G', 'GS', 'PTS', 'MP', 'PER', 'FTr', 'OWS', 'DWS', 'WS', 'VORP', 'FG', 'FGA', '3P', '3PA','2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB','TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']]
    data_cor = datafloat.corr()
    data_cor = data_cor['Salary']
    data_cor = data_cor.drop(['Salary'])
    data_cor_sum = data_cor.sum()
    data_cor_rate = 1 / data_cor_sum * data_cor
    stats = datafloat.drop(columns=['Salary'])
    stats_sum = stats.sum().astype(float)
    stats_cor_rate = stats / stats_sum * data_cor_rate
    player_co_rate = stats_cor_rate.sum(axis=1)
    salary = 100000000
    t["Player Co Rate"] = player_co_rate
    se = salary * player_co_rate
    se = se.astype(int)
    t["Salary Expected"] = se
    t["Difference"] = t["Salary Expected"] - t["Salary"]
    t["Difference"] = t["Difference"].astype(int)
    t.to_csv('CSVS/team{}.csv'.format(team), index=False)
    return t

def team_csv():
    for x in set(data["Team"]):
        team(x)
    