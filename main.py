import pygame
from pygame.color import THECOLORS
import pdb
import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import from_pygame, to_pygame
from pymunk.pygame_util import DrawOptions as draw
import pymunk.util as u
import random
import torch
import math
import numpy as np

# from MakeItLearn import *

width = 600  # Width Of The Game Window
height = 600  # Height Of The Game Window
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()  # Game Clock
SummarySensorData = []
StepSizeValue = 1 / 20.0  # Step Size For Simulation
ClockTickValue = 100  # Clock Tick
BotSpeed = 25  # Speed Of The Bot
model = None

def PointsFromAngle(angle):
    ### Returns The Unit Direction Vector With Given Angle ###
    return math.cos(angle), math.sin(angle)


class BotEnv:
    def __init__(self):
        ## Initialize Required Variables
        self.crashed = False
        self.DetectCrash = 0
        self.space = pymunk.Space()
        self.BuildBot(50.01, 450.01, 20)
        self.walls = []
        self.WallShapes = []
        self.WallRects = []
        ## Add Walls To The Map ###
        WallBody, WallShape, WallRect = self.BuildWall(200, 50, 50)
        self.WallRects.append(WallRect)
        WallBody, WallShape, WallRect = self.BuildWall(200, 125, 50)
        self.WallRects.append(WallRect)
        WallBody, WallShape, WallRect = self.BuildWall(200, 550, 50)
        self.WallRects.append(WallRect)
        WallBody, WallShape, WallRect = self.BuildWall(200, 450, 50)
        self.WallRects.append(WallRect)
        WallBody, WallShape, WallRect = self.BuildWall(400, 350, 50)
        self.WallRects.append(WallRect)
        WallBody, WallShape, WallRect = self.BuildWall(400, 250, 50)
        self.WallRects.append(WallRect)
        WallBody, WallShape, WallRect = self.BuildWall(500, 250, 50)
        self.WallRects.append(WallRect)
        WallBody, WallShape, WallRect = self.BuildWall(600, 250, 50)
        self.WallRects.append(WallRect)
        WallBody, WallShape, WallRect = self.BuildWall(115, 1050, 500)
        self.WallRects.append(WallRect)
        self.PreviousAction = 0

    def BuildWall(self, x, y, r):
        ### Build Wall On The Map ###
        size = r
        WallRect = pygame.Rect(x - r, 600 - y - r, 2 * r, 2 * r)
        return WallRect, WallRect, WallRect

    def BuildBot(self, x, y, r):
        ### Build The Bot Object ###
        size = r
        BoxPoints = list([Vec2d(-size, -size), Vec2d(-size, size), Vec2d(size, size), Vec2d(size, -size)])
        # BoxPoints = list(map(Vec2d, [(-size, -size), (-size, size), (size,size), (size, -size)]))
        mass = 0.5
        moment = pymunk.moment_for_poly(mass, BoxPoints, Vec2d(0, 0))
        self.Bot = pymunk.Body(mass, moment)
        self.Bot.position = (x, y)  # Declare Bot Position
        self.Bot.angle = 1.54  # Set the Bot Angle
        BotDirection = Vec2d(*PointsFromAngle(self.Bot.angle))  # Get The Direction Vector From Angle
        self.space.add(self.Bot)
        self.BotRect = pygame.Rect(x - r, 600 - y - r, 2 * r, 2 * r)
        return self.Bot

    def DrawEverything(self, flag=0):
        ### Write Everything On The Game Window ###
        img = pygame.image.load("./assets/intel.jpg")
        x, y = 580, 550
        AdjustedImagePosition = (x - 50, y + 50)
        screen.blit(img, to_pygame(AdjustedImagePosition, screen))
        if (flag == 0 and self.DetectCrash == 0):
            (self.BotRect.x, self.BotRect.y) = self.Bot.position[0], 600 - self.Bot.position[1]
            self.CircleRect = pygame.draw.circle(screen, (169, 169, 169), (self.BotRect.x, self.BotRect.y), 20, 0)
        ## If Collision Detected Draw Green
        elif (flag == 0 and self.DetectCrash >= 1):
            (self.BotRect.x, self.BotRect.y) = self.Bot.position[0], 600 - self.Bot.position[1]
            self.CircleRect = pygame.draw.circle(screen, (0, 255, 0), (self.BotRect.x, self.BotRect.y), 20, 0)
        ## If Collision Happens Draw Red
        else:
            (self.BotRect.x, self.BotRect.y) = self.Bot.position[0], 600 - self.Bot.position[1]
            self.CircleRect = pygame.draw.circle(screen, (255, 0, 0), (self.BotRect.x, self.BotRect.y), 20, 0)
        img = pygame.image.load("./assets/spherelight.png")
        offset = Vec2d(*img.get_size()) / 2.0
        x, y = self.Bot.position
        y = 600.0 - y
        AdjustedImagePosition = (x, y) - offset
        screen.blit(img, AdjustedImagePosition)
        for ob in self.WallRects:
            pygame.draw.rect(screen, (169, 169, 169), ob)

    def PlanAngle(self, A, B):
        ### Find The Angle Between Two Vector ###
        angle = np.arctan2(B[1] - A[1], B[0] - A[0])
        return angle

    def _step(self, action, CrashStep=0):
        ### Take The Simulation One Step Further ###
        self.Bot.angle = self.Bot.angle % 6.2831853072
        ### If Action Is Left
        if action == 3:
            self.Bot.angle -= 0.02
            self.PreviousBodyAngle = self.Bot.angle
            self.BotDirection = Vec2d(*PointsFromAngle(self.Bot.angle))
            BotDirection = self.BotDirection
            if (CrashStep > 0):
                self.Bot.velocity = BotSpeed / 3 * BotDirection
            else:
                self.Bot.velocity = BotSpeed * BotDirection
            self.PreviousAction = 3
        ### If Action Is Right
        elif action == 4:
            self.Bot.angle += 0.02
            self.PreviousBodyAngle = self.Bot.angle
            self.BotDirection = Vec2d(*PointsFromAngle(self.Bot.angle))
            BotDirection = self.BotDirection
            self.Bot.velocity = BotSpeed * BotDirection
            if (CrashStep == 1):
                self.Bot.velocity = BotSpeed / 3 * BotDirection
            else:
                self.Bot.velocity = BotSpeed * BotDirection
            self.PreviousAction = 4
        ### If Action Is Straight
        elif action == 5:
            self.Bot.angle += 0.
            self.PreviousBodyAngle = self.Bot.angle
            self.BotDirection = Vec2d(*PointsFromAngle(self.Bot.angle))
            BotDirection = self.BotDirection
            self.Bot.velocity = BotSpeed * BotDirection
            if (CrashStep == 1):
                self.Bot.velocity = BotSpeed / 3 * BotDirection
            else:
                self.Bot.velocity = BotSpeed * BotDirection

        screen.fill(THECOLORS["white"])  ## Clear The Screen
        self.DrawEverything()  ## Write Everything To The Game
        self.space.step(StepSizeValue)  ## Take One Step In Simulation
        clock.tick(ClockTickValue)  ## Tick The Clock
        x, y = self.Bot.position  ## Get The Bot Position
        SensorsData = self.AllSensorSensorsData(x, y, self.Bot.angle)  ## Get All The Sensor Data
        NormalizedSensorsData = [(x - 100.0) / 100.0 for x in SensorsData]  ## Normalize The Sensor Values
        state = np.array([NormalizedSensorsData])
        SensorsData = np.append(SensorsData, math.degrees(self.Bot.angle))
        SensorsData = np.append(SensorsData, [0])
        print(SensorsData[:-2])  ## Print The Sensor Data
        DataTensor = torch.Tensor(SensorsData[:-1]).view(1, -1)
        for ob in self.WallRects:
            if ob.colliderect(self.CircleRect):
                self.RecoverFromCrash(BotDirection)
        if (x >= 580 or x <= 20 or y <= 20 or y >= 680):
            self.RecoverFromCrash(BotDirection)
        SignalData = SensorsData[:-2]
        if 1 in SignalData:
            if (action == 5):
                action = self.PreviousAction
            self.crashed = True
            SensorsData[-1] = 1
            SummarySensorData.append(SensorsData)
            print(SensorsData[:-2])
            reward = -500
            self.RecoverFromCrash(BotDirection)
        else:
            self.DetectCrash = 0
            SummarySensorData.append(SensorsData)
        return

    def RecoverFromCrash(self, BotDirection):
        ### Execute Following When Bot Crashes ###
        while self.crashed:
            self.crashed = False
            for i in range(1):
                self.Bot.angle += 3.14
                self.BotDirection = Vec2d(*PointsFromAngle(self.Bot.angle))
                BotDirection = self.BotDirection
                self.Bot.velocity = BotSpeed * BotDirection
                screen.fill(THECOLORS["white"])
                self.DrawEverything(flag=1)
                self.space.step(StepSizeValue)
                pygame.display.flip()
                clock.tick(ClockTickValue)

    def AllSensorSensorsData(self, x, y, angle):
        ### Return The All Sensor Values ###
        SensorsData = []
        MiddleSensorStartPoint = (25 + x, y)
        MiddleSensorEndPoint = (65 + x, y)
        NumberOfSensors = 5
        RelativeAngles = []
        AngleToBeginWith = 1.3
        OffsetIncrement = (AngleToBeginWith * 2) / (NumberOfSensors - 1)
        RelativeAngles.append(-AngleToBeginWith)
        ## Generate Sensors
        for i in range(NumberOfSensors - 1):
            RelativeAngles.append(RelativeAngles[i] + OffsetIncrement)
        SensorList = []
        ## Collect The Sensor Value From All Sensors
        for i in range(NumberOfSensors):
            SensorList.append([MiddleSensorStartPoint, MiddleSensorEndPoint, RelativeAngles[i]])
            SensorsData.append(self.SensorReading(SensorList[i], x, y, angle))
        pygame.display.update()
        return SensorsData

    def SensorReading(self, sensor, x, y, angle):
        ### Returns The Reading For A Single Sensor ###
        distance = 0
        (x1, y1) = sensor[0][0], sensor[0][1]
        (x2, y2) = sensor[1][0], sensor[1][1]
        SensorAngle = sensor[2]
        PixelsInPath = []
        NumberOfPoints = 100
        ## Generate Sensor Points
        for k in range(NumberOfPoints):
            x_new = x1 + (x2 - x1) * (k / NumberOfPoints)
            y_new = y1 + (y2 - y1) * (k / NumberOfPoints)
            PixelsInPath.append((x_new, y_new))
        for pixel in PixelsInPath:
            distance += 1
            PixelInGame = self.Rotate((x, y), (pixel[0], pixel[1]), angle + SensorAngle)
            SensorStartInGame = self.Rotate((x, y), (x1, PixelsInPath[-1][1]), angle + SensorAngle)
            SensorEndInGame = self.Rotate((x, y), PixelsInPath[-1], angle + SensorAngle)
            if PixelInGame[0] <= 0 or PixelInGame[1] <= 0 or PixelInGame[0] >= width or PixelInGame[1] >= height:
                return distance
            else:
                for ob in self.WallRects:
                    if ob.collidepoint((PixelInGame[0], PixelInGame[1])):
                        return distance
        ## Draw The Sensor
        pygame.draw.line(screen, (30, 144, 255), SensorStartInGame, SensorEndInGame)
        return distance

    def Rotate(self, origin, point, angle):
        ### Rotates A Point Along A Given Point ###
        x1, y1 = origin
        x2, y2 = point
        final_x = x1 + math.cos(angle) * (x2 - x1) - math.sin(angle) * (y2 - y1)
        final_y = y1 + math.sin(angle) * (x2 - x1) + math.cos(angle) * (y2 - y1)
        final_y = abs(width - final_y)
        return final_x, final_y


def TakeLeftOrRightTurn(env):
    ### Take Random Action - Left Or Right Turn ###
    x = random.randint(3, 4)
    for i in range(40):
        env._step(x)


def GoStraight(env):
    ### Take Action To Go Straight ###
    x = random.randint(5, 5)
    for i in range(40):
        env._step(x)


if __name__ == "__main__":
    env = BotEnv()
    random.seed(10)
    env._step(3)
    for i in range(200):
        if (random.random() > 0.5):
            TakeLeftOrRightTurn(env)
        else:
            GoStraight(env)
        np.savetxt("./SensorData/SensorData.txt", SummarySensorData)
