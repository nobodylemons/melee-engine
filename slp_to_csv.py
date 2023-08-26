
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
    csvPath = os.path.join(DATA_DIR, filename.replace('.slp', '.csv'))
    # if os.path.exists(csvPath):
    #     return
    game = Game(os.path.join(SLP_DIR, filename))
    data = []
    portnums = []
    for j, p in enumerate(game.frames[0].ports):
        if p is None:
            continue
        portnums.append(j)
        
    for i, f in enumerate(game.frames):
        if i == 0:
            continue
        features = {}
        features['s_stage'] = str(game.start.stage)
        features['index'] = f.index
        # if not game.start.stage == id.Stage.FINAL_DESTINATION:
        #     return
        pnum = 0
        prev_f = game.frames[i-1]
        # prev_f2 = game.frames[i-2]
        
        for n in portnums:
            
#             post = prev_f2.ports[n].leader.post
            
#             features['s_player{}_position_x2'.format(pnum)] = myfloat(post.position.x)/255                    # Regression range 0-255?
#             features['s_player{}_position_y2'.format(pnum)] = myfloat(post.position.y)/255                    # Regression range 0-255?
            
            post = prev_f.ports[n].leader.post 
            # print(str(post.state))
            pred = f.ports[n].leader.post
            features[f'pred_player{pnum}_state'] = str(pred.state)
            features[f's_player{pnum}_state'] = str(post.state)                              # Multi category
            

            
            features[f's_player{pnum}_airborne'] = myfloat(post.airborne)                        # Binary category
            features[f's_player{pnum}_character'] = str(post.character)                      # Multi category
            features[f's_player{pnum}_combo_count'] = myfloat(post.combo_count)                  # Integer regression
            features[f's_player{pnum}_damage'] = myfloat(post.damage)/1000                            # Regression in range 0 to 999
            features[f's_player{pnum}_direction'] = myfloat(post.direction)/2 + .5                      # Binary category
            # features[f's_player{pnum}_flags'] = myfloat(post.flags)                              # Multi category (12)
            features[f's_player{pnum}_ground'] = str(post.ground)                            # Multi Category (ID)
            features[f's_player{pnum}_hit_stun'] = myfloat(post.hit_stun)                        # Regression range 0..
            features[f's_player{pnum}_jumps'] = myfloat(post.jumps)/2                              # Regression range 0..
            features[f's_player{pnum}_l_cancel'] = myfloat(post.l_cancel)                        # Binary category
            features[f's_player{pnum}_last_attack_landed'] = myfloat(post.last_attack_landed)    # Regression range 0..
            features[f's_player{pnum}_last_hit_by'] = str(post.last_hit_by)                  # Multi cateogry
            features[f's_player{pnum}_position_x'] = myfloat(post.position.x)/512 + .5                    # Regression range 0-255?
            features[f's_player{pnum}_position_y'] = myfloat(post.position.y)/512 + .5                    # Regression range 0-255?
            features[f's_player{pnum}_shield'] = myfloat(post.shield)/60                            # Regression range 0-255?
            
            features[f's_player{pnum}_state_age'] = myfloat(post.state_age)                      # Regression range 0..
            features[f's_player{pnum}_stocks'] = myfloat(post.stocks)/4                            # Regression range 0..
            
            pre = f.ports[n].leader.pre
            buttonVals = str(pre.buttons.physical).split('.')[1].split('|')
            for but in buttons:
                key = 'a_player{}_buttons_physical_{}'.format(pnum, but)
                features[key] = 0
            for but in buttonVals:
                # features['a_player{}_buttons_logical_{}'.format(pnum, str(but))] = 1
                features['a_player{}_buttons_physical_{}'.format(pnum, str(but))] = 1 #pre.buttons.physical

            features[f'a_player{pnum}_cstick_x'] = myfloat(pre.cstick.x)/2 + .5
            features[f'a_player{pnum}_cstick_y'] = myfloat(pre.cstick.y)/2 + .5
            # features['a_player{}_damage'.format(pnum)] = pre.damage
            # features['a_player{}_direction'.format(pnum)] = pre.direction
            features[f'a_player{pnum}_joystick_x'] = pre.joystick.x/2 + .5
            features[f'a_player{pnum}_joystick_y'] = pre.joystick.y/2 + .5
            # features['a_player{}_position_x'.format(pnum)] = myfloat(pre.position.x)
            # features['a_player{}_position_y'.format(pnum)] = myfloat(pre.position.y)
            # features['a_player{}_random_seed'.format(pnum)] = myfloat(pre.random_seed)
            # features['a_player{}_raw_analog_x'.format(pnum)] = pre.raw_analog_x
            # features['a_player{}_state'.format(pnum)] = myfloat(pre.state)
            features[f'a_player{pnum}_triggers_logical'] = myfloat(pre.triggers.logical)
            # features['a_player{}_triggers_physical_l'.format(pnum)] = pre.triggers.physical.l
            # features['a_player{}_triggers_physical_r'.format(pnum)] = pre.triggers.physical.r
            
            pnum += 1
        data.append(features)
        if pnum > 2:
            return
        
    df = pd.DataFrame(data).fillna(0)
    df.to_csv(csvPath,index=False, header=True)
    

if __name__ == "__main__":
    unprocessedFiles = []
    for filename in os.listdir(SLP_DIR):
        if not 'slp' in filename:
            continue
        if filename.count("Fox") < 2:
            continue
        # if "14_24_55" not in filename:
        #     continue
        unprocessedFiles.append(filename)

    from multiprocess import Pool
    import tqdm
    # processFile(unprocessedFiles[0])
    # print(unprocessedFiles)
    pool = Pool(processes=12)
    for _ in tqdm.tqdm(pool.imap_unordered(processFile, unprocessedFiles), total=len(unprocessedFiles)):
        pass
