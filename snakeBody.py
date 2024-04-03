class Snake_Body:
    def __init__(self, x, y, behind):
        self.x = x
        self.y = y
        self.behind = behind

    def moveHead(self, xDir, yDir):
        self.x += xDir
        self.y -= yDir

        if self.behind != None:
           return self.behind.moveBody(self.x - xDir, self.y + yDir)

    def moveBody(self, x, y):
        if self.behind != None:
            lastPostion = self.behind.moveBody(self.x, self.y)
        else:
            lastPostion = [ self.x, self.y ]

        self.x = x
        self.y = y
        return lastPostion

        