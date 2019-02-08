import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_stats = pd.read_csv("nba-players-stats/Seasons_Stats.csv")
data_salary = pd.read_csv("nba-players-stats/NBA_season1718_salary.csv")
data_players = pd.read_csv("nba-players-stats/Players.csv")
pd.set_option('display.max_columns', 50)

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

#barchart
def barcharteam():
# data to plot
    n_groups = len(data["Player"])
    salary = data["Salary"]
    exsalary = data["Salary Expected"]
 
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8
    rects1 = plt.bar(index, salary, bar_width, alpha=opacity, color='b',label='Salary')
    rects2 = plt.bar(index + bar_width, exsalary, bar_width, alpha=opacity, color='g', label='Salary Expected')

    plt.xlabel('Player')
    plt.ylabel('Salary')
    plt.title('Scores by person')
    plt.xticks(index + bar_width, list(range(len(data["Player"]))))
    plt.legend()

    plt.tight_layout()
    return plt.show()

def choose_team():
    print("TEAMS: MIN, PHO, MIL, GSW, SAS, BRK, DEN, DAL, POR, ORL, IND, PHI, DET, BOS, CHO, OKC, MEM, LAC, ATL, HOU, WAS, NOP, SAC, NYK, CHI, LAL, TOR, UTA, CLE, MIA")
    team = input("From what Team do you want the information? ")
    print('\n')
    data = pd.read_csv("CSVS/team{}.csv".format(team.upper()))
    return data

def choose_player(data):
    print("Write 'ALL' for all the information or player name: {} for a specific player ".format(list(data["Player"])))
    print('\n')
    player = input("Do you want all the information about the team or a specific player? ").upper()
    if player == "ALL" or player == "all":
        print(data)
        barcharteam()
    else:
        print(data.loc[data['Player'] == player])
       
    
def printteamerror():  
    try:
        data = choose_team()
        return data
        pass
    except:
        print("TIENES QUE ELEGIR ENTRE UNO DE LOS EQUIPOS DE LA LIGA.")
        printteamerror()
        
        
def printplayererror(data):
    try:
        choose_player(data)
        pass
    except:
        print("NOMBRE NO ENCONTRADO. ESTAS SON LAS ESTADÍSTICAS DE TODO EL EQUIPO")
        printplayererror(data)
    
if __name__ == "__main__":
    data = printteamerror()
    printplayererror(data)
    
  
