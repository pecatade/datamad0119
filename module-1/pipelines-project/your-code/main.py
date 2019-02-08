import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    barcharteam()
else:
    print(data.loc[data['Player'] == player])