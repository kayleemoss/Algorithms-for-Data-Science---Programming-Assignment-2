# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 08:45:20 2019

@author: Samuel Fisher, Intern
Johns Hopkins University Applied Physics Laboratory
"""

#Display who won and add to win counter
def whoWin(x,End,Xwin,Owin): 
    Xwin = 0
    Owin = 0
    if x == 1:
        End.configure(text="Player 1 has won!", background = 'white')
        Xwin = 1
    elif x == 2:
        End.configure(text="Player 2 has won!", background = 'white')
        Owin = 1
    else:
        End.configure(text="Nobody Wins", background = 'white')
    gameover = 1
    L = [Xwin,Owin,gameover]
    return L

#Check if there is a three in a row
#If there is a win, a display which team one and count that win
def checkWin(place,AIturn,End,Xwin,Owin,turn, aiSkill): 
    if place[1] == place[0] and place[0] == place[2] and place[1] != 0:
        print ("Player",place[1]," wins")
        return whoWin(place[1],End,Xwin,Owin)
    if place[0] == place[3] and place[0] == place[6] and place[0] != 0:
        print ("Player",place[0]," wins")
        return whoWin(place[0],End,Xwin,Owin)
    if place[0] == place[4] and place[0] == place[8] and place[0] != 0:
        print ("Player",place[0]," wins")
        return whoWin(place[0],End,Xwin,Owin)
    if place[1] == place[4] and place[1] == place[7] and place[1] != 0:
        print ("Player",place[1]," wins")
        return whoWin(place[1],End,Xwin,Owin)
    if place[2] == place[4] and place[2] == place[6] and place[2] != 0:
        print ("Player",place[2]," wins")
        return whoWin(place[2],End,Xwin,Owin)
    if place[2] == place[5] and place[2] == place[8] and place[2] != 0:
        print ("Player",place[2]," wins")
        return whoWin(place[2],End,Xwin,Owin)
    if place[3] == place[4] and place[3] == place[5] and place[3] != 0:
        print ("Player",place[3]," wins")
        return whoWin(place[3],End,Xwin,Owin)
    if place[6] == place[7] and place[8] == place[6] and place[6] != 0:
        print ("Player",place[6]," wins")
        return whoWin(place[7],End,Xwin,Owin)
    tie = 1
    for i in place:
        if i == 0:
            tie = 0
    if tie == 1:
        return whoWin(3,End,Xwin,Owin)
        
    return [0,0,0]

#Check who won without calling whoWin
#Necessary for MiniMax
def checkWin2(place):
    if place[1] == place[0] and place[0] == place[2] and place[1] != 0:
        return place[1]
    if place[0] == place[3] and place[0] == place[6] and place[0] != 0:
        return place[0]
    if place[0] == place[4] and place[0] == place[8] and place[0] != 0:
        return place[0]
    if place[1] == place[4] and place[1] == place[7] and place[1] != 0:
        return place[1]
    if place[2] == place[4] and place[2] == place[6] and place[2] != 0:
        return place[2]
    if place[2] == place[5] and place[2] == place[8] and place[2] != 0:
        return place[2]
    if place[3] == place[4] and place[3] == place[5] and place[3] != 0:
        return place[3]
    if place[6] == place[7] and place[8] == place[6] and place[6] != 0:
        return place[6]
    tie = 1
    for i in place:
        if i == 0:
            tie = 0
    if tie == 1:
        return 0
        
    return [0,0,0]

#Check possibilities for wins in the next move
# Goal based agent here
def checkWinPos(place):
    # check at 0-1-2
   if place[0] == 0 and place[1] == place[2] and place[2] != 0:
       return 0
   if place[1] == 0 and place[0] == place[2] and place[2] != 0:
       return 1
   if place[2] == 0 and place[0] == place[1] and place[1] != 0:
       return 2

    # check at 3-4-5
   if place[3] == 0 and place[4] == place[5] and place[5] != 0:
       return 3
   if place[4] == 0 and place[3] == place[5] and place[5] != 0:
       return 4
   if place[5] == 0 and place[3] == place[4] and place[4] != 0:
       return 5

    # check at 6-7-8
   if place[6] == 0 and place[7] == place[8] and place[8] != 0:
       return 6
   if place[7] == 0 and place[6] == place[8] and place[8] != 0:
       return 7
   if place[8] == 0 and place[6] == place[7] and place[7] != 0:
       return 8

    # check at 0-3-6
   if place[0] == 0 and place[3] == place[6] and place[6] != 0:
       return 0
   if place[3] == 0 and place[0] == place[6] and place[6] != 0:
       return 3
   if place[6] == 0 and place[0] == place[3] and place[3] != 0:
       return 6

    # check at 1-4-7
   if place[1] == 0 and place[4] == place[7] and place[7] != 0:
       return 1
   if place[4] == 0 and place[1] == place[7] and place[7] != 0:
       return 4
   if place[7] == 0 and place[1] == place[4] and place[4] != 0:
       return 7

    # check at 2-5-8
   if place[2] == 0 and place[5] == place[8] and place[8] != 0:
       return 2
   if place[5] == 0 and place[2] == place[8] and place[8] != 0:
       return 5
   if place[8] == 0 and place[2] == place[5] and place[5] != 0:
       return 8

    # check at 0-4-8
   if place[0] == 0 and place[4] == place[8] and place[8] != 0:
       return 0
   if place[4] == 0 and place[0] == place[8] and place[8] != 0:
       return 4
   if place[8] == 0 and place[0] == place[4] and place[4] != 0:
       return 8

    # check at 2-4-6
   if place[2] == 0 and place[4] == place[6] and place[6] != 0:
       return 2
   if place[4] == 0 and place[2] == place[6] and place[6] != 0:
       return 4
   if place[6] == 0 and place[2] == place[4] and place[4] != 0:
       return 6
