from Game_Parts.snakeBody import Snake_Body
import numpy as np

class Snake:
    def __init__(self, startX, startY, bodySize, xDir, yDir):
        self.bodySize = bodySize
        self.body = [ Snake_Body(startX, startY, None) ]

        self.xDir = xDir
        self.yDir = yDir

        self.directionQueue = []

        self.lastPostion = [None, None]

        for i in range(1, bodySize):
            new_part = Snake_Body(startX-i, startY, None)
            self.body.append(new_part)
            self.body[i-1].behind = new_part
        
    def changeDirection(self, turn):
        if turn[0] == 0:
            current_direction = (self.xDir, self.yDir)
            
            left_rotation = {(1, 0): (0, 1), (0, 1): (-1, 0), (-1, 0): (0, -1), (0, -1): (1, 0)}
            right_rotation = {(1, 0): (0, -1), (0, -1): (-1, 0), (-1, 0): (0, 1), (0, 1): (1, 0)}
            
            if turn[1] == 1:
                new_direction = right_rotation[current_direction]
            elif turn[2] == 1:
                new_direction = left_rotation[current_direction]
            
            self.xDir, self.yDir = new_direction
    
    def move(self, gridSize, applePos):
        self.lastPostion = self.body[0].moveHead(self.xDir, self.yDir)

        ateApple = self.body[0].x == applePos[0] and self.body[0].y == applePos[1]
        deadState = self.isDead(gridSize, body=self.body)

        if ateApple: 
            self._grow()
        
        return deadState

    def _grow(self):
        new_part = Snake_Body( self.lastPostion[0],  self.lastPostion[1], None)
        self.body[-1].behind = new_part
        self.body.append(new_part)
        self.bodySize += 1
    
    @staticmethod
    def isDead(gridSize, *, body, pt=None):
        if pt is None:
            pt = [body[0].x, body[0].y]

        if pt[0] > gridSize-1 or pt[1] > gridSize-1 or pt[0] < 0 or pt[1] < 0:
            return 1
        
        if pt in [[body[i].x, body[i].y] for i in range(1, len(body))]:
            return 1
        
        return 0
