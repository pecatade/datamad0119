import pandas as pd
pd.set_option('display.max_columns', 500)
print("TEAMS: MIN, PHO, MIL, GSW, SAS, BRK, DEN, DAL, POR, ORL, IND, PHI, DET, BOS, CHO, OKC, MEM, LAC, ATL, HOU, WAS, NOP, SAC, NYK, CHI, LAL, TOR, UTA, CLE, MIA")
team = input("From what Team do you want the information? ")
print('\n')
data = pd.read_csv("CSVS/team{}.csv".format(team))
print("Write 'ALL' for all the information or player name: {} for a specific player ".format(list(data["Player"])))
print('\n')
player = input("Do you want all the information about the team or a specific player? ")
if player == "ALL":
    print(data)
else:
    print(data.loc[data['Player'] == player])