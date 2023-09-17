SLP_DIR = '/home/robert/slippi_data/Slippi_Public_Dataset_v3'
DATA_DIR = '/root/training_data_csv'


CATEGORICAL_FEATURES = ['s_stage', 
                       
                       's_player0_state',
                       's_player0_character',
                     #   's_player0_flags',
                       's_player0_ground',
                       # # 'a_player0_buttons_logical',
                       # 'a_player0_state',

                       's_player1_state',
                       's_player1_character',
                     #   's_player1_flags',
                       's_player1_ground',
                       # 'a_player1_buttons_logical',
                       # 'a_player1_state'
                      ]

LABELS = ['pred_player0_state']