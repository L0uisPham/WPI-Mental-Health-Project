''' Features to use in RFC Analysis '''


from constants import *


runs_all = {
    'PHQ: Q': 
        (PHQ9_Q_NAMES,
        'PHQ 9 Risk Binary'),
    'GAD: Q': 
        (GAD7_Q_NAMES,
        'GAD 7 Risk Binary'),
    'PHQ: Q_Core': 
        (PHQ9_Q_NAMES + CORE_FEATURES,
        'PHQ 9 Risk Binary'),
    'GAD: Q_Core': 
        (GAD7_Q_NAMES + CORE_FEATURES,
        'GAD 7 Risk Binary'),
    'PHQ: Q_Core_GPA': 
        (PHQ9_Q_NAMES + CORE_FEATURES + ['GPA Weighted'],
        'PHQ 9 Risk Binary'),
    'GAD: Q_Core_GPA': 
        (GAD7_Q_NAMES + CORE_FEATURES + ['GPA Weighted'],
        'GAD 7 Risk Binary'),
    'PHQ: Core_GPA': 
        (CORE_FEATURES + ['GPA Weighted'],
        'PHQ 9 Risk Binary'),
    'GAD: Core_GPA': 
        (CORE_FEATURES + ['GPA Weighted'],
        'GAD 7 Risk Binary'),
    'PHQ: Core_GPA': 
        (CORE_FEATURES + ['GPA Weighted'],
        'PHQ 9 Risk Binary'),
    'GAD: Core': 
        (CORE_FEATURES + ['GPA Weighted'],
        'GAD 7 Risk Binary'),
    'PHQ: Core': 
        (CORE_FEATURES + ['GPA Weighted'],
        'PHQ 9 Risk Binary'),
    'Endorse Q9: Core_GPA':
        (CORE_FEATURES + ['GPA Weighted'],
        'Endorse Q9 Binary'),
}


question_level_final = {
    'PHQ: Q_Core_GPA': 
        (GAD7_Q_NAMES + PHQ9_Q_NAMES + CORE_FEATURES + ['GPA Weighted'],
        'PHQ 9 Risk Binary'),
    'GAD: Q_Core_GPA': 
        (GAD7_Q_NAMES + PHQ9_Q_NAMES + CORE_FEATURES + ['GPA Weighted'],
        'GAD 7 Risk Binary'),
    'Endorse Q9: Q_Core_GPA': 
        (GAD7_Q_NAMES + [q for q in PHQ9_Q_NAMES if '-9' not in q]+ CORE_FEATURES + ['GPA Weighted'],
        'Endorse Q9 Binary'),
}

summary_level_final = {
    'PHQ: Q_Core_GPA': 
        (['GAD 7 Risk Binary', 'Endorse Q9 Binary'] + EDU_FEATURES,
        'PHQ 9 Risk Binary'),
    'GAD: Q_Core_GPA': 
        (['PHQ 9 Risk Binary', 'Endorse Q9 Binary'] + EDU_FEATURES,
        'GAD 7 Risk Binary'),
    'Endorse Q9: Q_Core_GPA': 
        (['PHQ 9 Risk Binary', 'GAD 7 Risk Binary'] + EDU_FEATURES,
        'Endorse Q9 Binary'),
}


RUN_CONFIGS = {
    'runs_all':runs_all,
    'gpa':{k:i for k,i in runs_all.items() if 'GPA'in k},
    'runs_question':question_level_final,
    'runs_summary':summary_level_final
}