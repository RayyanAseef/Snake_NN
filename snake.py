from snakeBody import Snake_Body
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
        isDead = self.isDead(gridSize)

        if ateApple: 
            self._grow()
        
        return ateApple, isDead

    def _grow(self):
        new_part = Snake_Body( self.lastPostion[0],  self.lastPostion[1], None)
        self.body[-1].behind = new_part
        self.body.append(new_part)
        self.bodySize += 1
    
    def isDead(self, gridSize, pt = None):
        if pt is None:
            pt = self.body[0]

        if pt.x > gridSize-1 or pt.y > gridSize-1 or pt.x < 0 or pt.y < 0:
            return True
        
        if [pt.x, pt.y] in [[self.body[i].x, self.body[i].y] for i in range(1, len(self.body))]:
            return True
        
        return False
