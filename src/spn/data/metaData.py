

def get_decNode(dataset_name):

    if dataset_name == 'Computer_Diagnostician':
        return ['Rework_Decision']
    elif dataset_name == 'Export_Textiles':
        return ['Export_Decision']
    elif dataset_name == 'Test_Strep':
        return ['Treatment_Decision', 'Test_Decision']
    elif dataset_name == 'LungCancer_Staging':
        return ['Treatment', 'CT', 'Mediastinoscopy']
    elif dataset_name == 'HIV_Screening':
        return ['Screen', 'Treat_Counsel']
    elif dataset_name == 'Powerplant_Airpollution':
        return ['Installation_Type', 'Strike_Intervention']
    elif dataset_name == 'FrozenLake':
        return [f'Action_{i}' for i in range(10)]
    elif dataset_name == 'Elevators':
        return [f'Action_{i}' for i in range(6)]
    else:
        print(dataset_name)


def get_utilityNode(dataset_name):

    if dataset_name == 'Computer_Diagnostician':
        return ['Rework_Cost']
    elif dataset_name == 'Export_Textiles':
        return ['Profit']
    elif dataset_name == 'Test_Strep':
        return ['QALE']
    elif dataset_name == 'LungCancer_Staging':
        return ['Life_expectancy']
    elif dataset_name == 'HIV_Screening':
        return ['QALE']
    elif dataset_name == 'Powerplant_Airpollution':
        return ['Additional_Cost']
    elif dataset_name == 'FrozenLake':
        return ['Reward']
    elif dataset_name == 'Elevators':
        return ['Reward']
    else:
        print(dataset_name)

def get_scope_vars(dataset_name):

    #returns a list of all variables in sequence of partial order excluding decison variables
    #e.g.
    # if dataset_name == 'Computer_Diagnostician':
    #     return ['System_State','Logic_board_fail', 'IO_board_fail', 'Rework_Outcome', 'Rework_Cost' ]

    partial_order = get_partial_order(dataset_name)
    decision_nodes = get_decNode(dataset_name)
    var_set = [var for var_set in partial_order for var in var_set]
    for d in decision_nodes:
        var_set.remove(d)
    return var_set

def get_feature_names(dataset_name):

    partial_order = get_partial_order(dataset_name)
    var_set = [var for var_set in partial_order for var in var_set]
    return var_set


def get_partial_order(dataset_name):

    if dataset_name == 'Computer_Diagnostician':
        partialOrder = [['IO_board_fail', 'Logic_board_fail'], ['System_State'], ['Rework_Decision'], ['Rework_Outcome', 'Rework_Cost' ]]
        return partialOrder
    if dataset_name == 'Export_Textiles':
        partialOrder = [['Export_Decision'], ['Economical_State'], ['Profit']]
        return partialOrder
    if dataset_name == 'Test_Strep':
        partialOrder = [['Test_Decision'],['Streptococcal_Infection', 'Test_Result'],['Treatment_Decision'],
                        ['Rheumatic_Heart_Disease', 'Die_from_Anaphylaxis', 'Days_with_sore_throat', 'QALE']]
        return partialOrder
    if dataset_name == 'LungCancer_Staging':
        #partialOrder = [['CT'],['CTResult', 'Mediastinal_Metastases'],['Mediastinoscopy'],
                               #['Mediastinoscopy_Result', 'Mediastinoscopy_death'], ['Treatment'], ['Treatment_Death', 'Life_expectancy' ]]
        #ltest1
        partialOrder = [['CT'],['Mediastinal_Metastases', 'CTResult'],['Mediastinoscopy'],['Mediastinoscopy_Result', 'Mediastinoscopy_death'], ['Treatment'], ['Treatment_Death', 'Life_expectancy' ]]
        #ltest2
        #partialOrder = [['CT'],['Mediastinal_Metastases', 'CTResult'],['Mediastinoscopy'],['Mediastinoscopy_death', 'Mediastinoscopy_Result'], ['Treatment'], ['Treatment_Death', 'Life_expectancy' ]]
        #ltest3
        #partialOrder = [['CT'],['Mediastinal_Metastases'], ['CTResult'],['Mediastinoscopy'],['Mediastinoscopy_Result', 'Mediastinoscopy_death'], ['Treatment'], ['Treatment_Death', 'Life_expectancy' ]]
        return partialOrder
    if dataset_name == 'HIV_Screening':
        #partialOrder = [['Screen'], ['HIV_Test_Result', 'HIV_Status'],['Treat_Counsel'],
                        #['Compliance_Medical_Therapy',	'Reduce_Risky_Behavior', 'QALE']]
        partialOrder = [['Screen'], ['HIV_Status', 'HIV_Test_Result'],['Treat_Counsel'],
                        ['Compliance_Medical_Therapy',	'Reduce_Risky_Behavior', 'QALE']]
        return partialOrder
    if dataset_name == 'Powerplant_Airpollution':
        partialOrder = [['Installation_Type'],['Coal_Worker_Strike'],['Strike_Intervention'],['Strike_Resolution','Additional_Cost']]
        return partialOrder
    if dataset_name == 'FrozenLake':
        partialOrder = list()
        for i in range(10):
            partialOrder += [[f'State_{i}'], [f'Action_{i}']]
        partialOrder += [['State_10', 'Reward']]
        return partialOrder
    if dataset_name == 'Elevators':
        partialOrder = list()
        for i in range(6):
            partialOrder += [[f'Elevator_at_Floor_{i}', f'Person_Waiting_{i}', f'Person_in_Elevator_Going_Up_{i}', 
                                f'Elevator_Direction_{i}', f'Elevator_Closed_{i}'], [f'Action_{i}']]
        partialOrder += [[f'Elevator_at_Floor_6', f'Person_Waiting_6', f'Person_in_Elevator_Going_Up_6', 
                                f'Elevator_Direction_6', f'Elevator_Closed_6', 'Reward']]
        return partialOrder
    else:
        print(dataset_name)


def get_feature_labels(dataset_name):

    if dataset_name == 'Computer_Diagnostician':                                     # 6 variables
       return  ['IBF', 'LBF', 'SS', 'RD', 'RO', 'RC']
    if dataset_name == 'Export_Textiles':                                            # 3 variables
        return ['ED', 'ES', 'Pr']
    if dataset_name == 'Test_Strep':                                                 # 8 variables
        return ['TD', 'SI', 'TR', 'TRD', 'RH', 'Dfa', "Dws", 'Q']
    if dataset_name == 'LungCancer_Staging':                                         # 9 variables
        return ['CT', 'MM', 'CTR', 'Ms', 'MsR', 'MsD', 'Tr', 'TD', 'LE']
    if dataset_name == 'HIV_Screening':                                              # 7 variables
        return ['Sc', 'HS', 'HTR', 'TC', 'CMT', 'RRB', 'Q']
    if dataset_name == 'Powerplant_Airpollution':                                    # 5 variables
        return ['IT', 'CWS', 'SI', 'SR', 'AC']
    if dataset_name == 'FrozenLake':                                                 # 22 variables
        features = list()
        for i in range(10):
            features += [f'S{i}', f'A{i}']
        features += ['S10', 'RW']
        return features
    if dataset_name == 'Elevators': 
        features = list()
        for i in range(8):
            features += [f'EF{i}', f'PW{i}', f'PU{i}', f'ED{i}', f'EC{i}',  f'A{i}']
        features += [f'EF6', f'PW6', f'PU6', f'ED6', f'EC6', 'RW']
        return features





