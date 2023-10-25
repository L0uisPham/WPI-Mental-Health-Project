''' Helper functions for depression/anxiety scores '''

def phq9_map(val, tol='soft_num', return_categories=False):
    rankings = {
        'default' : [
            ('Mild', lambda x: x in range(0,10)),
            ('Moderate', lambda x: x in range(10,15)),
            ('Moderately_Severe', lambda x: x in range(15,20)),
            ('Severe', lambda x: x >=20) ##in range(20,28))
        ],
        'soft_num' : [
            (0, lambda x: x in range(0,10)),
            (1, lambda x: x >= 10)
        ],
        'soft_cat' : [
            ('Low', lambda x: x in range(0,10)),
            ('High', lambda x: x >= 10)
        ]
    }
    
    #return names of the categories
    if return_categories:
        return [x[0] for x in rankings[tol]]

    #get specific severity corresponding to input value
    for severity, is_in in rankings[tol]:
        if is_in(val):
            return severity
    
    print(f'Error, {val} not found in PHQ9 range.')

    return None
    

def gad7_map(val, tol='soft_num', return_categories=False):
    rankings = {
        'default' : [
            ('Minimal', lambda x: x in range(0,5)),
            ('Mild', lambda x: x in range(5,10)),
            ('Moderate', lambda x: x in range(10,15)),
            ('Severe', lambda x: x >= 15)
        ],
        'soft_num' : [
            (0, lambda x: x in range(0,10)),
            (1, lambda x: x >= 10)
        ],
        'soft_cat' : [ #NOTE: Used by anova to be categorical input
            ('Low', lambda x: x in range(0,10)),
            ('High', lambda x: x >= 10)
        ]
    }
    #return names of the categories
    if return_categories:
        return [x[0] for x in rankings[tol]]

    #get specific severity corresponding to input value
    for severity, is_in in rankings[tol]:
        if is_in(val):
            return severity

    print(f'Error, {val} not found in GAD7 range.')
    
    return None
    
def endorseQ9_map(val, tol='soft_num', return_categories=False):
    rankings = {
        'default' : [
            #TODO: Figure out what these values mean
            (0,lambda x: x==0),
            (1,lambda x: x==1),
            (2,lambda x: x==2),
            (3,lambda x: x==3),
        ],
        'soft_num' : [
            (0, lambda x: x == 0),
            (1, lambda x: x > 0)
        ],
        'soft_cat' : [ #NOTE: Used by anova to be categorical input
            ('Low', lambda x: x == 0),
            ('High', lambda x: x > 0)
        ]
    }
    #return names of the categories
    if return_categories:
        return [x[0] for x in rankings[tol]]

    #get specific severity corresponding to input value

    for severity, is_in in rankings[tol]:
        if is_in(val):
            return severity

    print(f'Error, {val} not found in GAD7 range.')
    
    return None

def q_map(val,tol='soft_num'):
    rankings = {
        'soft_num' : [
            (0, lambda x: x in [1,2]),
            (1, lambda x: x in [3,4]),
        ],
        'soft_cat' : [ #NOTE: Used by anova to be categorical input
            ('Absent', [1,2]),
            ('Present', [3,4])
        ]
    }

    for severity, is_in in rankings[tol]:
        if is_in(val):
            return severity

    print(f'Error, {val} not found in GAD7 range.')
    
    return None


sped_map = lambda x: {'Yes':1, 'No':0}[x]
planstatus_map = lambda x: {'Yes':1, 'No':0}[x]
ell_map = lambda x: {'Yes':1, 'No':0}[x]



