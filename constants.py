""" Feature configs for iterating and saving models """
#Non-MH Features: ['Gender', 'Grade', 'Race / Ethnicity', 'SPED status', '504 Plan Status', 'ELL status']

from filtering import FEATURES_DF

TARGET_MAPS = {
    'Endorse Q9': lambda x: endorseQ9_map(x),
    'PHQ 9 Risk': lambda x: phq9_map(x),
    'GAD 7 Risk': lambda x: gad7_map(x),
}

CORE_FEATURES = ['Gender', 'Race / Ethnicity', 'SPED status', '504 Plan Status', 'ELL status']
GAD7_Q_NAMES = [f'GAD7-{x+1}' for x in range(7)]
PHQ9_Q_NAMES = [f'PHQ9-{x+1}' for x in range(9)]
EDU_FEATURES = ['GPA Weighted','SPED status', '504 Plan Status', 'ELL status']
