import time
import pickle
import random
import matplotlib.pyplot as plt

#*---Global Declarables---*#
SLEEP = 0.008
STEP = 1000
FILENAME = "tictactoe/boardData.p"

x = 0
o = 0
draw = 0


average_reward_q1 : list[float] = [0]
average_reward_v1 : list[float] = [0]

#*---AI Declarables---*#
num_actions = 9

#*---AI Hyperparameters---*#
learning_rate = 0.1         # alpha: step size for Q-value updates (0 < alpha <= 1)
discount_factor = 0.99      # gamma: future reward discount factor (0 < gamma <= 1)
epsilon1 = 0.9              # epsilon: probability of random action (exploration rate)
epsilon2 = 0.9              # epsilon: probability of random action (exploration rate)
epsilon_min = 0.01          # minimum value for epsilon (stops decaying at this value)
epsilon_decay = 0.999       # decay rate for epsilon after each episode (0 < decay <= 1)
num_episodes = 150000 + 1   # total number of training episodes

Q_table = {}
V_table = {}

#*---Player Classes---*#
class player:
    def __init__(self, c: str):
        self.p = c

    def makeMove(self, game_instance) -> None:
        index = -1
        while not 0 <= index < 9:
            setCsr(0, 11)
            try:
                index = int(input("Please enter the index number (1 - 9)|\t")) - 1
            except Exception:
                index = -1
            if not 0 <= index < 9:
                print("Error Bad Data")
            if not game_instance.placePiece(index):
                print("Space Taken")
                index = -1

class Qai(player):
    def makeMove(self, game_instance) -> None:
        state_str = game_instance.boardToString(game_instance.board)
        available_actions = game_instance.getAvailableSpaces(game_instance.board)
        if state_str in Q_table and any(Q_table[state_str]):
            q_values = [Q_table[state_str][a] for a in available_actions]
            max_q = max(q_values)
            best_actions = [a for a in available_actions if Q_table[state_str][a] == max_q]
            action = random.choice(best_actions)
        else:
            action = random.choice(available_actions)
        game_instance.placePiece(action)

class RandomAgent(player):
    def makeMove(self, game_instance) -> None:
        available_actions = game_instance.getAvailableSpaces(game_instance.board)
        action = random.choice(available_actions)
        game_instance.placePiece(action)

class Vai(player):
    def makeMove(self, game_instance) -> None:
        #*---Value Agent move selection (greedy)---*#
        state_str = game_instance.boardToString(game_instance.board)
        available_actions = game_instance.getAvailableSpaces(game_instance.board)

        # Evaluate each possible move by the value of the resulting state
        best_value = float('-inf')
        best_actions = []
        for action in available_actions:
            # Simulate the move
            temp_board = game_instance.board.copy()
            temp_board[action] = self.p
            next_state_str = game_instance.boardToString(temp_board)
            value = V_table.get(next_state_str, 0.0)
            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)
        action = random.choice(best_actions)
        game_instance.placePiece(action)

#*---Game Class---*#
class game:
    def __init__(self, ptype1: str = "q", ptype2: str = "v"):
        self.board = [" "] * 9
        self.pTurn = "H"
        if ptype1 == "v":
            self.player1 = Vai("x")
        elif ptype1 == "q":
            self.player1 = Qai("x")
        else:
            self.player1 = player("x")
        
        if ptype2 == "v":
            self.player2 = Vai("o")
        elif ptype2 == "q":
            self.player2 = Qai("o")
        else:
            self.player2 = player("o")
        

    def clearBoard(self) -> None:
        for i in range(len(self.board)):
            self.board[i] = " "

    def boardToString(self, board) -> str:
        return ''.join(['x' if c == 'x' else 'o' if c == 'o' else '.' for c in board])

    def printBoard(self) -> None:
        for j in range(3):
            setColr(37, 40)
            print("+---+---+---+\n|", end = "")
            for i in range(3):
                index = i + j * 3
                if self.board[index] == " ":
                    setColr(37, 40) 
                    print(f" {index + 1} ", end = "")
                    setColr(37, 40)
                    print("|", end = "")
                elif self.board[index] == self.player1.p:
                    setColr(37, 42)
                    print(f" {self.player1.p} ", end = "")
                    setColr(37, 40)
                    print("|", end = "")
                elif self.board[index] == self.player2.p:
                    setColr(37, 41)
                    print(f" {self.player2.p} ", end = "")
                    setColr(37, 40)
                    print("|", end = "")
            print()
        setColr(37, 40)
        print("+---+---+---+")

    def win(self, pTurn) -> bool:
        for i in range(3):
            if self.board[i] == pTurn and self.board[i + 3] == pTurn and self.board[i + 6] == pTurn:
                return True
        for j in range(3):
            if self.board[j * 3] == pTurn and self.board[j * 3 + 1] == pTurn and self.board[j * 3 + 2] == pTurn:
                return True
        if self.board[0] == pTurn and self.board[4] == pTurn and self.board[8] == pTurn:
            return True
        if self.board[2] == pTurn and self.board[4] == pTurn and self.board[6] == pTurn:
            return True
        return False

    def getAvailableSpaces(self, board) -> list:
        return [i for i in range(9) if board[i] == " "]

    def placePiece(self, index: int) -> bool:
        if self.board[index] == " ":
            self.board[index] = self.pTurn
            return True
        else:
            return False

    def weGame(self) -> str:
        global x, o, draw
        clearScreen()
        self.clearBoard()
        currPlayer = self.player1
        loop = True
        while loop:
            clearScreen()
            self.pTurn = currPlayer.p

            setCsr(0, 0)
            print(f"Turn:\t {self.pTurn}")

            setCsr(0, 3)
            self.printBoard()

            setCsr(0, 15)
            scoreBoard(x, o, draw)

            currPlayer.makeMove(self)

            if self.win(self.pTurn):
                loop = False
                clearScreen()
                setCsr(0, 0)
                print(f"Turn:")
                setCsr(0, 3)
                self.printBoard()
                setCsr(0, 15)
                scoreBoard(x, o, draw)
                time.sleep(SLEEP)
                return currPlayer.p
            
            elif len(self.getAvailableSpaces(self.board)) == 0:
                loop = False
                clearScreen()
                setCsr(0, 0)
                print("Turn:")
                setCsr(0, 3)
                self.printBoard()
                setCsr(0, 15)
                scoreBoard(x, o, draw)
                time.sleep(SLEEP)
                return "Draw"
            currPlayer = self.player2 if currPlayer == self.player1 else self.player1
            time.sleep(SLEEP)

    def aiReset(self):
        self.clearBoard()
        return self.boardToString(self.board)

    def aiStep(self, action: int, agent: player | Qai | Vai) -> list:
        self.placePiece(action)
        if self.win(self.pTurn) and self.pTurn == agent.p:
            return self.boardToString(self.board), 10, True
        elif self.win(self.pTurn) and self.pTurn != agent.p:
            return self.boardToString(self.board), -10, True
        elif len(self.getAvailableSpaces(self.board)) == 0:
            return self.boardToString(self.board), 0, True
        else:
            return self.boardToString(self.board), 0, False  # or -0.01
    
    def trainQAi(self):
        global epsilon1, Q_table, average_reward_q1
        epsilon1 = 0.9
        self.player1 = Qai("x")
        self.player2 = RandomAgent("o")
        interval_reward_p1 = 0
        interval_reward_p2 = 0
        for episode in range(num_episodes):
            self.aiReset()
            done = False
            currPlayer = self.player1
            self.pTurn = currPlayer.p
            state_str = self.boardToString(self.board)
            while not done:
                available_actions = self.getAvailableSpaces(self.board)
                if currPlayer == self.player1:
                    if state_str not in Q_table:
                        Q_table[state_str] = [0.0] * num_actions
                    if random.random() < epsilon1:
                        action = random.choice(available_actions)
                    else:
                        q_values = [Q_table[state_str][a] for a in available_actions]
                        max_q = max(q_values)
                        best_actions = [a for a in available_actions if Q_table[state_str][a] == max_q]
                        action = random.choice(best_actions)
                    next_board, reward, terminated = self.aiStep(action, currPlayer)
                    done = terminated
                    next_state_str = self.boardToString(self.board)
                    if next_state_str not in Q_table:
                        Q_table[next_state_str] = [0.0] * num_actions
                    Q_table[state_str][action] += learning_rate * (
                        reward + discount_factor * max(Q_table[next_state_str]) - Q_table[state_str][action]
                    )
                else:
                    currPlayer.makeMove(self)

                # Assign reward before switching players
                if currPlayer.p == "x":
                    interval_reward_p1 += reward

                state_str = next_state_str
                currPlayer = self.player2 if currPlayer == self.player1 else self.player1
                self.pTurn = currPlayer.p

                if done:
                    target = reward
                else:
                    target = reward + discount_factor * max(Q_table[next_state_str])
                Q_table[state_str][action] += learning_rate * (target - Q_table[state_str][action])

            epsilon1 = max(epsilon_min, epsilon1 * epsilon_decay)
            if (episode + 1) % STEP == 0:
                average_reward_q1.append(interval_reward_p1 / STEP)
                interval_reward_p1 = 0
                print(f"Episode {episode + 1}/{num_episodes - 1}, Avg Reward P1: {average_reward_q1[-1]:.2f}")
    
    def trainVAi(self):
        global epsilon2, V_table, average_reward_v1
        epsilon2 = 0.9
        self.player1 = Vai("x")
        self.player2 = RandomAgent("o")
        interval_reward_p1 = 0
        for episode in range(num_episodes):
            self.aiReset()
            done = False
            currPlayer = self.player1
            self.pTurn = currPlayer.p
            state_str = self.boardToString(self.board)
            while not done:
                available_actions = self.getAvailableSpaces(self.board)
                if state_str not in V_table:
                    V_table[state_str] = 0.0
                if random.random() < epsilon2:
                    action = random.choice(available_actions)
                else:
                    best_value = float('-inf')
                    best_actions = []
                    for a in available_actions:
                        temp_board = self.board.copy()
                        temp_board[a] = currPlayer.p
                        next_state_str = self.boardToString(temp_board)
                        if next_state_str not in V_table:
                            V_table[next_state_str] = 0.0
                        value = V_table.get(next_state_str, 0.0)
                        if value > best_value:
                            best_value = value
                            best_actions = [a]
                        elif value == best_value:
                            best_actions.append(a)
                    action = random.choice(best_actions)
                next_board, reward, terminated = self.aiStep(action, currPlayer)
                done = terminated
                next_state_str = self.boardToString(self.board)
                if next_state_str not in V_table:
                    V_table[next_state_str] = 0.0
                V_table[state_str] += learning_rate * (reward + discount_factor * V_table[next_state_str] - V_table[state_str])
                # Assign reward before switching players
                if currPlayer.p == "x":
                    interval_reward_p1 += reward

                state_str = next_state_str
                currPlayer = self.player2 if currPlayer == self.player1 else self.player1
                self.pTurn = currPlayer.p

                if done:
                    target = reward
                else:
                    target = reward + discount_factor * V_table[next_state_str]
                V_table[state_str] += learning_rate * (target - V_table[state_str])

            epsilon2 = max(epsilon_min, epsilon2 * epsilon_decay)
            if (episode + 1) % STEP == 0:
                average_reward_v1.append(interval_reward_p1 / STEP)
                interval_reward_p1 = 0
                print(f"Episode {episode + 1}/{num_episodes - 1}, Avg Reward P1: {average_reward_v1[-1]:.2f}")

    def agentComp(self, qagent_rewards, vagent_rewards, episodes, filename="reward_comparison.png", label_q="QAgent", label_v="VAgent"):
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, qagent_rewards, label=label_q, color='blue')
        plt.plot(episodes, vagent_rewards, label=label_v, color='green')
        plt.xlabel("Episodes")
        plt.ylabel("Average Reward")
        plt.title("Average Reward per Episode: QAgent vs VAgent")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Graph saved as {filename}")


#*---Terminal Manipulation---*#
def clearScreen():
    print("\033c\033[3J")

def clearLine():
    print("\033[2K")

def setCsr(x: int = 0, y: int = 0):
    print(f"\033[{y};{x}H")

def saveCsr():
    print("\033[7")

def resetCsr():
    print("\033[8")

def setColr(fColr: int = 37, gColr: int = 40):
    print(f"\033[{fColr};{gColr}m", end="")

def scoreBoard(x, o, draw):
    print("+----------+-------+")
    print("| {:<8} | {:<5} |".format("Player 1", x))
    print("| {:<8} | {:<5} |".format("Player 2", o))
    print("| {:<8} | {:<5} |".format("Draw", draw))
    print("+----------+-------+")

#*---Main Program---*#
if __name__ == "__main__":
    while True:
        menu = int(input("\n1. Play\n2. Train\n3. Exit\n"))
        p1 : str
        p2 : str
        if menu == 1:
            menu2 = int(input("\n1. Player\n2. QAgent\n3. VAgent\n"))
            if menu2 == 1:
                p1 = 'p'
            elif menu2 == 2:
                p1 = 'q'
            elif menu2 == 3:
                p1 = 'v'
            else:
                continue
            
            menu2 = int(input("\n1. Player\n2. QAgent\n3. VAgent\n"))
            if menu2 == 1:
                p2 = 'p'
            elif menu2 == 2:
                p2 = 'q'
            elif menu2 == 3:
                p2 = 'v'
            else:
                continue
            
            game_instance = game(p1, p2)
            
            try:
                data = pickle.load(open(FILENAME, 'rb'))
                Q_table = data['q1']
                V_table = data['v1']
            except:
                Q_table.clear()
                V_table.clear()
                game_instance.trainQAi()
                game_instance.trainVAi()
                game_instance.agentComp(average_reward_q1, average_reward_v1, range(0, num_episodes, STEP), filename="/home/strawberry/Documents/Programming/Python/tictactoe/agent_rewards1.png", label_q="QAgent", label_v="VAgent")

                data = {'q1': Q_table, 'v1': V_table}
                pickle.dump(data, open(FILENAME, 'wb'))

            while True:
                win = game_instance.weGame()
                if win == "x":
                    x += 1
                elif win == "o":
                    o += 1
                elif win == "Draw":
                    draw += 1
        elif menu == 2:
            Q_table.clear()
            V_table.clear()
            game_instance = game('q', 'q')
            game_instance.trainQAi()
            game_instance.trainVAi()
            game_instance.agentComp(average_reward_q1, average_reward_v1, range(0, num_episodes, STEP), filename="/home/strawberry/Documents/Programming/Python/tictactoe/agent_rewards1.png", label_q="QAgent", label_v="VAgent")

            data = {'q1': Q_table, 'v1': V_table}
            pickle.dump(data, open(FILENAME, 'wb'))
        else:
            exit(0)