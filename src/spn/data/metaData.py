

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
    elif dataset_name == 'Navigation':
        return [f'Action_{i}' for i in range(5)]
    elif dataset_name == 'CrossingTraffic':
        return [f'Action_{i}' for i in range(5)]
    elif dataset_name == 'SkillTeaching':
        return [f'Action_{i}' for i in range(5)]
    elif dataset_name == 'GameOfLife':
        return [f'Action_{i}' for i in range(3)]
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
    elif dataset_name == 'Navigation' or dataset_name == 'CrossingTraffic':
        return ['Reward']
    elif dataset_name == 'GameOfLife':
        return ['Reward']
    elif dataset_name == 'SkillTeaching':
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
        #partialOrder = [['CT'],['CTResult', 'Mediastinal_Metastases'],['Mediastinoscopy'],['Mediastinoscopy_death', 'Mediastinoscopy_Result'], ['Treatment'], ['Treatment_Death', 'Life_expectancy' ]]
        #partialOrder = [['CT'],['CTResult', 'Mediastinal_Metastases'],['Mediastinoscopy'],['Mediastinoscopy_Result', 'Mediastinoscopy_death'], ['Treatment'], ['Treatment_Death', 'Life_expectancy' ]]
        #ltest1
        partialOrder = [['CT'],['Mediastinal_Metastases', 'CTResult'],['Mediastinoscopy'],['Mediastinoscopy_Result', 'Mediastinoscopy_death'], ['Treatment'], ['Treatment_Death', 'Life_expectancy' ]]
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
    if dataset_name == 'Navigation':
        partialOrder = list()
        for i in range(5):
            partialOrder += [[f'Robot_at_1_t{i}', f'Robot_at_2_t{i}', f'Robot_at_3_t{i}', 
                                f'Robot_at_4_t{i}', f'Robot_at_5_t{i}', f'Robot_at_6_t{i}'], [f'Action_{i}']]
        partialOrder += [[f'Robot_at_1_t5', f'Robot_at_2_t5', f'Robot_at_3_t5', 
                                f'Robot_at_4_t5', f'Robot_at_5_t5', f'Robot_at_6_t5', 'Reward']]
        return partialOrder
    if dataset_name == 'CrossingTraffic':
        partialOrder = list()
        for i in range(5):
            partialOrder += [[f'Robot_at_1_t{i}', f'Robot_at_2_t{i}', f'Robot_at_3_t{i}', 
                                f'Robot_at_4_t{i}', f'Robot_at_5_t{i}', f'Robot_at_6_t{i}',
                                f'Robot_at_7_t{i}', f'Robot_at_8_t{i}', f'Robot_at_9_t{i}',
                                f'Obstacle_at_2_t{i}', f'Obstacle_at_5_t{i}', f'Obstacle_at_8_t{i}'], [f'Action_{i}']]
        partialOrder += [[f'Robot_at_1_t5', f'Robot_at_2_t5', f'Robot_at_3_t5', 
                                f'Robot_at_4_t5', f'Robot_at_5_t5', f'Robot_at_6_t5',
                                f'Robot_at_7_t5', f'Robot_at_8_t5', f'Robot_at_9_t5',
                                f'Obstacle_at_2_t5', f'Obstacle_at_5_t5', f'Obstacle_at_8_t5', 'Reward']]
        return partialOrder
    if dataset_name == 'GameOfLife':
        partialOrder = list()
        for i in range(3):
            partialOrder += [[f'Cell_1_t{i}', f'Cell_2_t{i}', f'Cell_3_t{i}', 
                                f'Cell_4_t{i}', f'Cell_5_t{i}', f'Cell_6_t{i}',
                                f'Cell_7_t{i}', f'Cell_8_t{i}', f'Cell_9_t{i}'], [f'Action_{i}']]
        partialOrder += [[f'Cell_1_t3', f'Cell_2_t3', f'Cell_3_t3', 
                                f'Cell_4_t3', f'Cell_5_t3', f'Cell_6_t3',
                                f'Cell_7_t3', f'Cell_8_t3', f'Cell_9_t3', 'Reward']]
        return partialOrder
    if dataset_name == 'SkillTeaching':
        partialOrder = list()
        for i in range(5):
            partialOrder += [[f'HintDelayVarS0_t{i}', f'HintDelayVarS1_t{i}', f'HintedRightS0_t{i}', f'HintedRightS1_t{i}',
                                f'ProficiencyMedS0_t{i}', f'ProficiencyMedS1_t{i}', f'UpdateTurnS0_t{i}', f'UpdateTurnS1_t{i}',
                                f'AnsweredRightS0_t{i}', f'AnsweredRightS1_t{i}', f'ProficiencyHighS0_t{i}', f'ProficiencyHighS1_t{i}'],
                                [f'Action_{i}']]
        partialOrder += [[f'HintDelayVarS0_t5', f'HintDelayVarS1_t5', f'HintedRightS0_t5', f'HintedRightS1_t5',
                                f'ProficiencyMedS0_t5', f'ProficiencyMedS1_t5', f'UpdateTurnS0_t5', f'UpdateTurnS1_t5',
                                f'AnsweredRightS0_t5', f'AnsweredRightS1_t5', f'ProficiencyHighS0_t5', f'ProficiencyHighS1_t5',
                                 'Reward']]
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
        for i in range(6):
            features += [f'EF{i}', f'PW{i}', f'PU{i}', f'ED{i}', f'EC{i}',  f'A{i}']
        features += [f'EF6', f'PW6', f'PU6', f'ED6', f'EC6', 'RW']
        return features
    if dataset_name == 'Navigation': 
        features = list()
        for i in range(5):
            features += [f'R1{i}', f'R2{i}', f'R3{i}', f'R4{i}',  f'R5{i}', f'R6{i}', f'A{i}']
        features += [f'R15', f'R25', f'R35', f'R45', f'R55', f'R65', 'RW']
        return features
    if dataset_name == 'GameOfLife': 
        features = list()
        for i in range(3):
            features += [f'C1{i}', f'C2{i}', f'C3{i}', f'C4{i}',  f'C5{i}', f'C6{i}', f'C7{i}',  f'C8{i}', f'C9{i}', f'A{i}']
        features += [f'C13', f'C23', f'C33', f'C43', f'C53', f'C63', f'C73', f'C83', f'C93', 'RW']
        return features
    if dataset_name == 'CrossingTraffic': 
        features = list()
        for i in range(5):
            features += [f'R1{i}', f'R2{i}', f'R3{i}', f'R4{i}', f'R5{i}', f'R6{i}',
                         f'R7{i}', f'R8{i}', f'R9{i}', f'O2{i}', f'O5{i}', f'O8{i}', f'A{i}']
        features += [f'R15', f'R25', f'R35', f'R45', f'R55', f'R65',
                     f'R75', f'R85', f'R95', f'O25', f'O55', f'O85', 'RW']
        return features
    if dataset_name == 'SkillTeaching': 
        features = list()
        for i in range(5):
            features += [f'HD0{i}', f'HD1{i}', f'HR0{i}', f'HR1{i}', f'PM0{i}', f'PM1{i}', f'UT0{i}', f'UT1{i}',
                                f'AR0{i}', f'AR1{i}', f'PH0{i}', f'PH1{i}', f'A{i}']
        features += [f'HD05', f'HD15', f'HR05', f'HR15', f'PM05', f'PM15', f'UT05', f'UT15',
                                f'AR05', f'AR15', f'PH05', f'PH15', 'RW']
        return features





