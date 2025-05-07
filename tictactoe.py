#*---Imports---*#
import time
import pickle
import random

x = 0
o = 0
draw = 0

#*---AI Creation---*#

#*---AI Declarables---*#
num_actions = 9

#*---AI Hyperparameters---*#
learning_rate = 0.2         # Alpha
discount_factor = 0.99      # Gamma
epsilon = 1.0               # Initial exploration rate
epsilon_min = 0.01          # Minimum exploration rate
epsilon_decay = 0.995       # Decay rate for epsilon
num_episodes = 200000         # Total number of episodes for training

# Initialize Q-table with small random values
#Q_table = [[random.uniform(0, 0.1) for _ in range(num_actions)] for _ in range(num_states)]
Q_table_p1 = {}  # For player 1 ('x')
Q_table_p2 = {}  # For player 2 ('o')

#*---Class Mess---*#
class player:
    p = "H"

    def __init__(self, c : str):
        self.p = c

    def makeMove(self, game_instance) -> None:
        #*---Local Declarables---*#
        index : int = -1

        #*---Take Input---*#            
        while not 0 <= index < 9:
            setCsr(0, 11)
            index : int = int(input("Please enter the index number (1 - 9)|\t")) - 1
            if not 0 <= index < 9:
                print("Error Bad Data")
              
            if not game_instance.placePiece(index):
                print("Space Taken")
                index = -1

class ai1(player):
    def makeMove(self, game_instance) -> None:
        state_str = game_instance.boardToString(game_instance.board)
        available_actions = game_instance.getAvailableSpaces(game_instance.board)

        # Use Q_table_p1 for player1 ('x')
        Q_table = Q_table_p1

        # Choose best action or random if not learned yet
        if state_str in Q_table and any(Q_table[state_str]):
            q_values = [Q_table[state_str][a] for a in available_actions]
            max_q = max(q_values)
            best_actions = [a for a in available_actions if Q_table[state_str][a] == max_q]
            action = random.choice(best_actions)
        else:
            action = random.choice(available_actions)
        game_instance.placePiece(action)

class ai2(player):
    def makeMove(self, game_instance) -> None:
        state_str = game_instance.boardToString(game_instance.board)
        available_actions = game_instance.getAvailableSpaces(game_instance.board)

        # Use Q_table_p2 for player2 ('o')
        Q_table = Q_table_p2

        # Choose best action or random if not learned yet
        if state_str in Q_table and any(Q_table[state_str]):
            q_values = [Q_table[state_str][a] for a in available_actions]
            max_q = max(q_values)
            best_actions = [a for a in available_actions if Q_table[state_str][a] == max_q]
            action = random.choice(best_actions)
        else:
            action = random.choice(available_actions)
        game_instance.placePiece(action)

class game:
    pTurn = "H"    

    def __init__(self, ptype1 : str = "a", ptype2 : str = "a"):
        self.board : list[9] = [" ", " ", " ", 
                                " ", " ", " ", 
                                " ", " ", " "]
        self.player1 = player("x") if ptype1 == "p" else ai1("x")
        self.player2 = player("o") if ptype2 == "p" else ai2("o")
        
        
    #*---Clear The Game Board---*#
    def clearBoard(self) -> None:
        for i in range(len(self.board)):
            self.board[i] = " "
    
    #*---Board To String---*#
    def boardToString(self, board) -> str:
        string : list = []
        for i in board:
            if i == " ":
                string.append(".")
            elif i == self.player1.p:
                string.append("0")
            elif i == self.player2.p:
                string.append("1")
        return "".join(string)

    #*---Print The Board to Terminal---*#
    def printBoard(self) -> None:
        for j in range(3):
            print("+---+---+---+\n|", end = "")
            for i in range(3):
                if self.board[i + j * 3] == " ":
                    print(f" {i + j * 3 + 1} |", end = "")
                elif self.board[i + j * 3] == self.player1.p:
                    print(f" {self.player1.p} |", end = "")
                elif self.board[i + j * 3] == self.player2.p:
                    print(f" {self.player2.p} |", end = "")
            print()
        print("+---+---+---+")


    #*---Has the Current Player Just Won?---*#
    def win(self, pTurn) -> bool:
        #*---Check Vertical Win---*#
        for i in range(3):
            if self.board[i] == pTurn and self.board[i + 3] == pTurn and self.board[i + 6] == pTurn:
                return True
        
        #*---Check Horizontal Win---*#
        for j in range(3):
            if self.board[j * 3] == pTurn and self.board[j * 3 + 1] == pTurn and self.board[j * 3 + 2] == pTurn:
                return True
    
        #*---Check Diagonal Win---*#
        if self.board[0] == pTurn and self.board[4] == pTurn and self.board[8] == pTurn:
            return True
    
        if self.board[2] == pTurn and self.board[4] == pTurn and self.board[6] == pTurn:
            return True
        
        #*---If Not Win---*#
        return False
    
    
    #*---Return a List of All Available Spaces---*#
    def getAvailableSpaces(self, board : list) -> list:
        available : list = []
        for i in range(9):
            if board[i] == " ":
                available.append(i)
        return available


    #*---Place a Piece---*#
    def placePiece(self, index : int) -> bool:
        if self.board[index] == " ":
            self.board[index] = self.pTurn
            return True

        else:
            return False


    #*---Begin the Game---*#
    def weGame(self) -> str:
        #*---Loop Declarables---*#
        scores : list[2] = [0, 0]

        

        #*---Clear Screen---*#
        clearScreen()
        self.clearBoard()

        #*---Game Loop---*#
        currPlayer : player = self.player1
        
        loop : bool = True
        while loop:
            #*---Loop Declarables---*#
            

            #*---Start New Round---*#
            clearScreen()
            
            #*---Set Display Character(e.g. 'x' 'o')---*#
            self.pTurn = currPlayer.p
            
            #*---Show Whose Turn It Is---*#
            setCsr(0, 0)
            print(f"Turn:\t {self.pTurn}")
            
            #*---Display The Board---*#
            setCsr(0, 3)
            self.printBoard()

            setCsr(0, 15)
            scoreBoard(x, o, draw)

            #*---Have Player/AI Play Their Turn---*#
            currPlayer.makeMove(self)

            #*---Winning Move?---*#
            if self.win(self.pTurn):
                loop = False
                clearScreen()
                self.printBoard()
                return currPlayer.p
            
            elif len(self.getAvailableSpaces(self.board)) == 0:
                loop = False
                clearScreen()
                self.printBoard()
                return "Draw"
            
            #*---Swap Turns---*#
            if currPlayer == self.player1:
                currPlayer = self.player2
            else:
                currPlayer = self.player1

            time.sleep(0.01)
            
    
    def aiReset(self):
        #*---Reset the Board---*#
        self.clearBoard()

        #*---Reset the Scores---*#
        scores : list[2] = [0, 0]

        #*---Reset the Game Loop---*#
        loop : bool = True
    
        return self.boardToString(self.board)


    def aiStep(self, action : int, agent : ai1 | ai2) -> list:
        #*---Take Action---*#
        self.placePiece(action)

        #*---Check if Game is Over---*#
        if self.win(self.pTurn) and self.pTurn == agent.p:
            return self.boardToString(self.board), 1, True, False
        elif self.win(self.pTurn) and self.pTurn != agent.p:
            return self.boardToString(self.board), -1, True, False
        elif len(self.getAvailableSpaces(self.board)) == 0:
            return self.boardToString(self.board), 0, True, False
        
        else:
            return self.boardToString(self.board), -0.1, False, False



    def trainAi(self):
        global epsilon
        for episode in range(num_episodes):
            self.aiReset()
            done = False
            total_reward_p1 = 0
            total_reward_p2 = 0
            currPlayer = self.player1
            self.pTurn = currPlayer.p
            state_str = self.boardToString(self.board)
            while not done:
                available_actions = self.getAvailableSpaces(self.board)
                # Select the correct Q-table
                Q_table = Q_table_p1 if currPlayer == self.player1 else Q_table_p2

                if random.random() < epsilon:
                    action = random.choice(available_actions)
                else:
                    q_values = [Q_table.setdefault(state_str, [0.0]*num_actions)[a] for a in available_actions]
                    max_q = max(q_values)
                    best_actions = [a for a in available_actions if Q_table[state_str][a] == max_q]
                    action = random.choice(best_actions)

                next_board, reward, terminated, truncated = self.aiStep(action, currPlayer)
                done = terminated or truncated
                next_state_str = self.boardToString(self.board)

                # Initialize Q-table entries if missing
                if state_str not in Q_table:
                    Q_table[state_str] = [0.0] * num_actions
                if next_state_str not in Q_table:
                    Q_table[next_state_str] = [0.0] * num_actions

                # Q-learning update for the current player
                Q_table[state_str][action] += learning_rate * (
                    reward + discount_factor * max(Q_table[next_state_str]) - Q_table[state_str][action]
                )

                # Track rewards separately
                if currPlayer == self.player1:
                    total_reward_p1 += reward
                else:
                    total_reward_p2 += reward

                state_str = next_state_str

                # Swap players and update pTurn
                currPlayer = self.player2 if currPlayer == self.player1 else self.player1
                self.pTurn = currPlayer.p

                if done:
                    break

            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{num_episodes}, Total Reward P1: {total_reward_p1}, Total Reward P2: {total_reward_p2}")

    def evaluateAgent(self, Q_table_p1, Q_table_p2, num_episodes=100):
        total_rewards_p1 = 0
        total_rewards_p2 = 0
        for episode in range(num_episodes):
            self.aiReset()
            done = False
            currPlayer = self.player1
            self.pTurn = currPlayer.p
            state_str = self.boardToString(self.board)
            while not done:
                available_actions = self.getAvailableSpaces(self.board)
                # Select the correct Q-table
                Q_table = Q_table_p1 if currPlayer == self.player1 else Q_table_p2

                q_values = [Q_table.setdefault(state_str, [0.0]*num_actions)[a] for a in available_actions]
                max_q = max(q_values)
                best_actions = [a for a in available_actions if Q_table[state_str][a] == max_q]
                action = random.choice(best_actions)

                next_board, reward, terminated, truncated = self.aiStep(action, currPlayer)
                done = terminated or truncated
                state_str = self.boardToString(self.board)

                # Track rewards separately
                if currPlayer == self.player1:
                    total_rewards_p1 += reward
                else:
                    total_rewards_p2 += reward

                # Swap players and update pTurn
                currPlayer = self.player2 if currPlayer == self.player1 else self.player1
                self.pTurn = currPlayer.p

                if done:
                    break
        average_reward_p1 = total_rewards_p1 / num_episodes
        average_reward_p2 = total_rewards_p2 / num_episodes
        print(f"Average reward for Player 1 (X) over {num_episodes} episodes: {average_reward_p1}")
        print(f"Average reward for Player 2 (O) over {num_episodes} episodes: {average_reward_p2}")
        return average_reward_p1, average_reward_p2
    
#*---Terminal Manipulation---*#
def clearScreen():
    print("\033c\033[3J")

def clearLine():
    print("\033[2K")

def setCsr(x : int = 0, y : int = 0):
    print(f"\033[{y};{x}H")

def saveCsr():
    print("\033 7")

def restCsr():
    print("\033 8")

def setColr(fColr : int = 37, gColr : int = 40):
    print(f"\033[{fColr};{gColr}m")


#*---Global Methods---*#
def scoreBoard(x, o, draw):
    print("+----------+-------+")
    print("| {:<8} | {:<5} |".format("Player 1", x))
    print("| {:<8} | {:<5} |".format("Player 2", o))
    print("| {:<8} | {:<5} |".format("Draw", draw))
    print("+----------+-------+")


#                                                                                                                                                                                       
#                                                                                                                                                                                       
#                 AAA                tttt               tttt                                                         tttt            iiii                                           !!! 
#                A:::A            ttt:::t            ttt:::t                                                      ttt:::t           i::::i                                         !!:!!
#               A:::::A           t:::::t            t:::::t                                                      t:::::t            iiii                                          !:::!
#              A:::::::A          t:::::t            t:::::t                                                      t:::::t                                                          !:::!
#             A:::::::::A   ttttttt:::::tttttttttttttt:::::ttttttt        eeeeeeeeeeee    nnnn  nnnnnnnn    ttttttt:::::ttttttt    iiiiiii    ooooooooooo   nnnn  nnnnnnnn         !:::!
#            A:::::A:::::A  t:::::::::::::::::tt:::::::::::::::::t      ee::::::::::::ee  n:::nn::::::::nn  t:::::::::::::::::t    i:::::i  oo:::::::::::oo n:::nn::::::::nn       !:::!
#           A:::::A A:::::A t:::::::::::::::::tt:::::::::::::::::t     e::::::eeeee:::::een::::::::::::::nn t:::::::::::::::::t     i::::i o:::::::::::::::on::::::::::::::nn      !:::!
#          A:::::A   A:::::Atttttt:::::::tttttttttttt:::::::tttttt    e::::::e     e:::::enn:::::::::::::::ntttttt:::::::tttttt     i::::i o:::::ooooo:::::onn:::::::::::::::n     !:::!
#         A:::::A     A:::::A     t:::::t            t:::::t          e:::::::eeeee::::::e  n:::::nnnn:::::n      t:::::t           i::::i o::::o     o::::o  n:::::nnnn:::::n     !:::!
#        A:::::AAAAAAAAA:::::A    t:::::t            t:::::t          e:::::::::::::::::e   n::::n    n::::n      t:::::t           i::::i o::::o     o::::o  n::::n    n::::n     !:::!
#       A:::::::::::::::::::::A   t:::::t            t:::::t          e::::::eeeeeeeeeee    n::::n    n::::n      t:::::t           i::::i o::::o     o::::o  n::::n    n::::n     !!:!!
#      A:::::AAAAAAAAAAAAA:::::A  t:::::t    tttttt  t:::::t    tttttte:::::::e             n::::n    n::::n      t:::::t    tttttt i::::i o::::o     o::::o  n::::n    n::::n      !!! 
#     A:::::A             A:::::A t::::::tttt:::::t  t::::::tttt:::::te::::::::e            n::::n    n::::n      t::::::tttt:::::ti::::::io:::::ooooo:::::o  n::::n    n::::n          
#    A:::::A               A:::::Att::::::::::::::t  tt::::::::::::::t e::::::::eeeeeeee    n::::n    n::::n      tt::::::::::::::ti::::::io:::::::::::::::o  n::::n    n::::n      !!! 
#   A:::::A                 A:::::A tt:::::::::::tt    tt:::::::::::tt  ee:::::::::::::e    n::::n    n::::n        tt:::::::::::tti::::::i oo:::::::::::oo   n::::n    n::::n     !!:!!
#  AAAAAAA                   AAAAAAA  ttttttttttt        ttttttttttt      eeeeeeeeeeeeee    nnnnnn    nnnnnn          ttttttttttt  iiiiiiii   ooooooooooo     nnnnnn    nnnnnn      !!! 
#                                                                                                                                                                                       
#                                                                                                                                                                                       
#                                                                                                                                                                                       
#                                                                                                                                                                                       
#                                                                                                                                                                                       
#                                                                                                                                                                                       
#                                                                                                                                                                                       
# font: doh
#
# also append the outcome of every game (board and score) to a file



#*---Main Program---*#
if __name__ == "__main__":
    scores : list[2]


    #*---Create Game---*#
    game_instance : game
    
    menu : int = int(input("1. Play\n2. Train\n3. Exit\n"))
    if menu == 1:
        menu2 : int = int(input("1. Player vs Player\n2. Player vs AI\n3. AI vs AI\n"))
        if menu2 == 1:
            game_instance = game('p', 'p')
        elif menu2 == 2:
            game_instance = game('p', 'a')
        elif menu2 == 3:
            game_instance = game('a', 'a')
        else:
            exit(0)
        
        try:
            data = pickle.load(open("boardData.p", 'rb'))
            Q_table_p1, Q_table_p2 = data
        except:
            data = []
            game_instance.trainAi()
            game_instance.evaluateAgent(Q_table_p1, Q_table_p2)
            data = Q_table_p1, Q_table_p2
            pickle.dump(data, open("boardData.p", 'wb'))
        
        #*---Begin Game---*#
        while True:
            win = game_instance.weGame()
            if win == "x":
                x += 1
            elif win == "o":
                o += 1
            elif win == "Draw":
                draw += 1

    elif menu == 2:
        game_instance.trainAi()
        game_instance.evaluateAgent(Q_table_p1, Q_table_p2)
    else:
        exit(0)
    
