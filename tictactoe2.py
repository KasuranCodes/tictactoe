with open("boardData.txt", 'w') as bd:
    data = {}
    data = pickle.load(bd)
    data[self.boardToString(self.board)] = qVal
    pickle.dump(data, bd)