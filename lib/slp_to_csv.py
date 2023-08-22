
import pickle 
import zlib
import pandas as pd
import json
from slippi import Game, id
import os
from lib.vars import *

def myfloat(n):
    if n is None:
        return -1
    return float(n)

def myfloat0(n):
    if n is None:
        return 0
    return float(n)

buttons = ['Y', 'R', 'B', 'L', 'Z', 'A', 'X', 'START', 'DPAD_UP', 'DPAD_RIGHT', 'DPAD_DOWN', 'DPAD_LEFT']

def processFile(filename):
    csvPath = os.path.join(data_dir, filename.replace('.slp', '.csv'))
    if os.path.exists(csvPath):
        return
    game = Game(os.path.join(slp_dir, filename))
    data = []
    portnums = []
    for i, f in enumerate(game.frames):
        
        if i == 0:
            for j, p in enumerate(f.ports):
                if p is None:
                    continue
                portnums.append(j)
            continue
        features = {}
        features['s_stage'] = myfloat(game.start.stage)
        if not game.start.stage == id.Stage.FINAL_DESTINATION:
            return
        pnum = 0
        prev_f = game.frames[i-1]
        # prev_f2 = game.frames[i-2]
        
        for n in portnums:
            
#             post = prev_f2.ports[n].leader.post
            
#             features['s_player{}_position_x2'.format(pnum)] = myfloat(post.position.x)/255                    # Regression range 0-255?
#             features['s_player{}_position_y2'.format(pnum)] = myfloat(post.position.y)/255                    # Regression range 0-255?
            
            post = prev_f.ports[n].leader.post
            

            
            features['s_player{}_airborne'.format(pnum)] = myfloat(post.airborne)                        # Binary category
            features['s_player{}_character'.format(pnum)] = myfloat(post.character)                      # Multi category
            features['s_player{}_combo_count'.format(pnum)] = myfloat(post.combo_count)                  # Integer regression
            features['s_player{}_damage'.format(pnum)] = myfloat(post.damage)/1000                            # Regression in range 0 to 999
            features['s_player{}_direction'.format(pnum)] = myfloat(post.direction)/2 + .5                      # Binary category
            features['s_player{}_flags'.format(pnum)] = myfloat(post.flags)                              # Multi category (12)
            features['s_player{}_ground'.format(pnum)] = myfloat(post.ground)                            # Multi Category (ID)
            features['s_player{}_hit_stun'.format(pnum)] = myfloat(post.hit_stun)                        # Regression range 0..
            features['s_player{}_jumps'.format(pnum)] = myfloat(post.jumps)/2                              # Regression range 0..
            features['s_player{}_l_cancel'.format(pnum)] = myfloat(post.l_cancel)                        # Binary category
            features['s_player{}_last_attack_landed'.format(pnum)] = myfloat(post.last_attack_landed)    # Regression range 0..
            features['s_player{}_last_hit_by'.format(pnum)] = myfloat(post.last_hit_by)                  # Multi cateogry
            features['s_player{}_position_x'.format(pnum)] = myfloat(post.position.x)/512 + .5                    # Regression range 0-255?
            features['s_player{}_position_y'.format(pnum)] = myfloat(post.position.y)/512 + .5                    # Regression range 0-255?
            features['s_player{}_shield'.format(pnum)] = myfloat(post.shield)/60                            # Regression range 0-255?
            features['s_player{}_state'.format(pnum)] = myfloat(post.state)                              # Multi category
            features['s_player{}_state_age'.format(pnum)] = myfloat(post.state_age)                      # Regression range 0..
            features['s_player{}_stocks'.format(pnum)] = myfloat(post.stocks)/4                            # Regression range 0..
            
            pre = f.ports[n].leader.pre
            buttonVals = str(pre.buttons.physical).split('.')[1].split('|')
            for but in buttons:
                key = 'a_player{}_buttons_physical_{}'.format(pnum, but)
                features[key] = 0
            for but in buttonVals:
                # features['a_player{}_buttons_logical_{}'.format(pnum, str(but))] = 1
                features['a_player{}_buttons_physical_{}'.format(pnum, str(but))] = 1 #pre.buttons.physical

            features['a_player{}_cstick_x'.format(pnum)] = myfloat(pre.cstick.x)/2 + .5
            features['a_player{}_cstick_y'.format(pnum)] = myfloat(pre.cstick.y)/2 + .5
            # features['a_player{}_damage'.format(pnum)] = pre.damage
            # features['a_player{}_direction'.format(pnum)] = pre.direction
            features['a_player{}_joystick_x'.format(pnum)] = pre.joystick.x/2 + .5
            features['a_player{}_joystick_y'.format(pnum)] = pre.joystick.y/2 + .5
            # features['a_player{}_position_x'.format(pnum)] = myfloat(pre.position.x)
            # features['a_player{}_position_y'.format(pnum)] = myfloat(pre.position.y)
            # features['a_player{}_random_seed'.format(pnum)] = myfloat(pre.random_seed)
            # features['a_player{}_raw_analog_x'.format(pnum)] = pre.raw_analog_x
            # features['a_player{}_state'.format(pnum)] = myfloat(pre.state)
            features['a_player{}_triggers_logical'.format(pnum)] = myfloat(pre.triggers.logical)
            # features['a_player{}_triggers_physical_l'.format(pnum)] = pre.triggers.physical.l
            # features['a_player{}_triggers_physical_r'.format(pnum)] = pre.triggers.physical.r
            
            pred = f.ports[n].leader.post
            features['pred_player{}_state'.format(pnum)] = myfloat0(pred.state)
            
            
            pnum += 1
            data.append(features)
        if pnum > 2:
            return
        
    df = pd.DataFrame(data).fillna(0)
    df.to_csv(csvPath,index=False, header=True)
    
