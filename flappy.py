from itertools import cycle
from collections import defaultdict
import random,sys,os,uuid,pickle,math
import numpy as np
import pygame
from pygame.locals import *
from multiprocessing import Process, Queue
from multiprocessing import Pool
from decimal import *
import itertools
import os

#os.environ["SDL_VIDEODRIVER"] = "dummy"

unique_filename = ''
global score
total_score = 0
score = 0
# Q / Reward

q = defaultdict(int)
#q = pickle.load(open('cloud_data/data/0.7_0.95_10000_20000_0.5033_.p', "rb"))


reward = 0.0
i = 0

exploration = 0

# Actions: 0 - Don't Flap, 1 - Flap
actions = [0,1]
history = []

# Initialize Previous Action
previous_action = 0
previous_state = 0
playerState = ()

# Game Environment Variables
FPS = 60
TRIAL = 0
PIPEGAPSIZE = 100
SCREENWIDTH = 288
SCREENHEIGHT = 512
PLAYERCLOCK = 0
BASEY = SCREENHEIGHT * 0.79

# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        # amount by which base can maximum shift to left
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)


def main(params):
    global alpha, gamma, epsilon , base, total_score,TRIAL,q
    total_score = 0
    base = 'data/' + str(uuid.uuid4()).split('-')[0]
    global SCREEN, FPSCLOCK, playerState, unique_filename
    alpha, gamma, epsilon  = params
    q_val = "_%s_%s_%s_" % (alpha, gamma, TRIAL)
    print q_val

    pygame.init()
    FPSCLOCK = pygame.time.Clock()

    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird')

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # game over sprite
    IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
    # message sprite for welcome screen
    IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    SOUNDS['die']    = pygame.mixer.Sound('assets/audio/die' + soundExt)
    SOUNDS['hit']    = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    SOUNDS['point']  = pygame.mixer.Sound('assets/audio/point' + soundExt)
    SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    SOUNDS['wing']   = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    while True:
        # select random background sprites
        randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

        # select random player sprites
        randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
        IMAGES['player'] = (
            pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )

        # select random pipe sprites
        pipeindex = random.randint(0, len(PIPES_LIST) - 1)
        IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), 180),
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hismask for pipes
        HITMASKS['pipe'] = (
            getHitmask(IMAGES['pipe'][0]),
            getHitmask(IMAGES['pipe'][1]),
        )

        # hitmask for player
        HITMASKS['player'] = (
            getHitmask(IMAGES['player'][0]),
            getHitmask(IMAGES['player'][1]),
            getHitmask(IMAGES['player'][2]),
        )

        movementInfo = showWelcomeAnimation()
        crashInfo = mainGame(movementInfo)
        showGameOverScreen(crashInfo)


def showWelcomeAnimation():
    """Shows welcome screen animation of flappy bird"""
    # index of player to blit on screen
    playerIndex = 0
    playerIndexGen = cycle([0, 1, 2, 1])
    # iterator used to change playerIndex after every 5th iteration
    loopIter = 0

    playerx = int(SCREENWIDTH * 0.2)
    playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)

    messagex = int((SCREENWIDTH - IMAGES['message'].get_width()) / 2)
    messagey = int(SCREENHEIGHT * 0.12)

    basex = 0
    # amount by which base can maximum shift to left
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # player shm for up-down motion on welcome screen
    playerShmVals = {'val': 0, 'dir': 1}

    while True:
        return {
            'playery': playery + playerShmVals['val'],
            'basex': basex,
            'playerIndexGen': playerIndexGen,
        }

        # adjust playery, playerIndex, basex
        if (loopIter + 1) % 8 == 0:
            playerIndex = playerIndexGen.next()
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 4) % baseShift)
        playerShm(playerShmVals)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))
        SCREEN.blit(IMAGES['player'][playerIndex],
                    (playerx, playery + playerShmVals['val']))
        SCREEN.blit(IMAGES['message'], (messagex, messagey))
        SCREEN.blit(IMAGES['base'], (basex, BASEY))

        pygame.display.update()
        FPSCLOCK.tick(FPS)



def mainGame(movementInfo):
    global playerHeight,playerState,reward,PLAYERCLOCK, score
    time_elapsed_since_last_action = 0
    clock = pygame.time.Clock()

    score = playerIndex = loopIter = 0
    playerIndexGen = movementInfo['playerIndexGen']
    playerHeight = IMAGES['player'][playerIndex].get_height()
    playerWidth = IMAGES['player'][0].get_width() / 2
    playerx, playery = int(SCREENWIDTH * 0.2), movementInfo['playery']
    basex = movementInfo['basex']
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    pipeVelX = -4

    # player velocity, max velocity, downward accleration, accleration on flap
    playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
    playerMaxVelY =  10   # max vel along Y, max descend speed
    playerMinVelY =  -8   # min vel along Y, max ascend speed
    playerAccY    =   1   # players downward accleration
    playerFlapAcc =  -9   # players speed on flapping
    playerFlapped = False # True when player flaps
    playerMidPos = playerx + IMAGES['player'][0].get_width() / 2


    while True:
        dt = clock.tick()
        time_elapsed_since_last_action += dt
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP) and time_elapsed_since_last_action > 100:
                time_elapsed_since_last_action = 0
                if playery > -2 * IMAGES['player'][0].get_height():
                    playerVelY = playerFlapAcc
                    playerFlapped = True
                    SOUNDS['wing'].play()

        aboveGround =  int(((SCREENHEIGHT - playery) - (SCREENHEIGHT - BASEY))/ (playerHeight * .5))
        nextPipeYs = [ p['y'] for p in lowerPipes if p['x'] > playerMidPos]
        nextPipeXs = [ p['x'] for p in lowerPipes if p['x'] > playerMidPos]
        nextPipeHeight =  int((SCREENHEIGHT - nextPipeYs[0]) / (playerHeight*.5))
        nextPipeDistance =  int((nextPipeXs[0] - playerMidPos) / (playerWidth*.5))

        playerState = (aboveGround,nextPipeDistance,nextPipeHeight,playerVelY/2)
        # check for crash here

        crashTest = checkCrash({'x': playerx, 'y': playery, 'index': playerIndex}, upperPipes, lowerPipes)

        if crashTest[0]:
            reward = 0
            return {
                'y': playery,
                'groundCrash': crashTest[1],
                'basex': basex,
                'upperPipes': upperPipes,
                'lowerPipes': lowerPipes,
                'score': score,
                'playerVelY': playerVelY,
            }


        # check for score
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                score += 1
                reward = 100
                lowerPipes = lowerPipes[-2:]
                #SOUNDS['point'].play()

        # playerIndex basex change
        if (loopIter + 1) % 3 == 0:
            playerIndex = playerIndexGen.next()
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        # player's movement
        if playerVelY < playerMaxVelY and not playerFlapped:
            playerVelY += playerAccY
        if playerFlapped:
            playerFlapped = False
        playerHeight = IMAGES['player'][playerIndex].get_height()
        playery += min(playerVelY, BASEY - playery - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))


        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        # show score so player overlaps the score
        showScore(score)
        SCREEN.blit(IMAGES['player'][playerIndex], (playerx, playery))
        pygame.display.update()
        FPSCLOCK.tick(FPS)

        if time_elapsed_since_last_action > 80:
            decision = act(playerState)
            time_elapsed_since_last_action = 0

            if decision == 0:
                pass
            else:
                if playery > 2 * IMAGES['player'][0].get_height():
                    playerVelY = playerFlapAcc
                    playerFlapped = True



def showGameOverScreen(crashInfo):
    global TRIAL, playerState,reward,history,alpha,gamma,epsilon,base, unique_filename, score, total_score, q_val
    total_score += score
    TRIAL += 1
    if TRIAL % 1000 == 0:
        avg_score = total_score / float(TRIAL)
        total_score = 0
        #unique_filename = 'data/' + str(alpha)+'_'+str(gamma)+'_'+str(epsilon)+'_'+ str(TRIAL) + '_' + str(avg_score) + '_.p'
        unique_filename =  q_val + '_' + str(avg_score) + '_.p'
        with open(unique_filename, 'wb') as f:
           pickle.dump(q,f)

    if exploration < 0.03:
        history = []
        FPSCLOCK.tick(FPS)
        pygame.display.update()
        return

def playerShm(playerShm):
    """oscillates the value of playerShm['val'] between 8 and -8"""
    if abs(playerShm['val']) == 8:
        playerShm['dir'] *= -1

    if playerShm['dir'] == 1:
         playerShm['val'] += 1
    else:
        playerShm['val'] -= 1


def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},  # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be shown

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    global reward,playerState
    """returxns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return [True, True]
    else:
        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return [True, False]

    return [False, False]

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

def getQ():
    path = 'data'
    filenames = next(os.walk('data'))[2]
    pAvg = defaultdict(int)

    for pick in filenames:
        pick = pickle.load(open(path+'/'+pick, "rb"))
        for k,v in pick.items():
            pAvg[k] += pick[k]

    return pAvg


def act(state):
    global q, actions, exploration,previous_action,reward,unique_filename,i, previous_state, history

    options = [ q[(state,act)] for act in actions ]
    best_option = max(options)
    best_action = actions[options.index(best_option)]
    #print ['%.16f' % n for n in options ], state, exploration
    print state, epsilon
    possible_actions = [ a for a in actions]
    possible_actions.append(best_action)

    action_weights = [(exploration * .5),(exploration * .5),(1 - exploration)]
    action = np.random.choice(possible_actions,p=action_weights)
    q[(previous_state, previous_action)] += alpha * (reward + (gamma * q[(state,action)]) - q[(previous_state, previous_action)])
    state_action = [(state, action)]
    history.append(state_action)

    previous_action = action
    previous_state = state
    exploration = math.exp(-TRIAL/epsilon)

    return action


if __name__ == '__main__':
    testing = False
    alphas = [ 0.25, 0.50, 0.70 ]
    gammas = [ 1.0, 0.95 ]
    epsilons = [ 2000000.0 ]
    params = [ alphas, gammas, epsilons]
    param_grid = list(itertools.product(*params))
    p = Pool(8)
    p.map(main,param_grid)

    # params = (0,0,20000,'cloud_data/data/0.7_0.95_10000_20000_0.5033_.p')
    #
    # if testing:
    #     import os
    #     scores = []
    #     cloud_data = next(os.walk('cloud_data/data'))[2]
    #     for q_values in cloud_data:
    #         params = (0,0,0,'cloud_data/data/'+q_values)
    #         score = main(params)
    #         line = q_values + 'SCORE:' + str(score)
    #         scores.append(line)
    #     print scores
    #
    # else:
    #     main(params)