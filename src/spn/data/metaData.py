

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
    else:
        print(dataset_name)


def get_utilityNode(dataset_name):

    if dataset_name == 'Computer_Diagnostician':
        return ['Rework_Cost']
    if dataset_name == 'Export_Textiles':
        return ['Profit']
    if dataset_name == 'Test_Strep':
        return ['QALE']
    if dataset_name == 'LungCancer_Staging':
        return ['Life_expectancy']
    if dataset_name == 'HIV_Screening':
        return ['QALE']
    if dataset_name == 'Powerplant_Airpollution':
        return ['Additional_Cost']

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
        #partialOrder = [['System_State'], ['Rework_Decision'], ['Logic_board_fail', 'IO_board_fail', 'Rework_Outcome', 'Rework_Cost' ]]
        partialOrder = [['Logic_board_fail', 'IO_board_fail','System_State'], ['Rework_Decision'], ['Rework_Outcome', 'Rework_Cost' ]]
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
        partialOrder = [['CT'],['Mediastinal_Metastases', 'CTResult'],['Mediastinoscopy'],
                               ['Mediastinoscopy_Result', 'Mediastinoscopy_death'], ['Treatment'], ['Treatment_Death', 'Life_expectancy' ]]
        return partialOrder
    if dataset_name == 'HIV_Screening':
        #partialOrder = [['Screen'], ['HIV_Test_Result', 'HIV_Status'],['Treat_Counsel'],
                        #['Compliance_Medical_Therapy',	'Reduce_Risky_Behavior', 'QALE']]
        partialOrder = [['Screen'], ['HIV_Status', 'HIV_Test_Result'],['Treat_Counsel'],
                        ['Compliance_Medical_Therapy',	'Reduce_Risky_Behavior', 'QALE']]
        return partialOrder
    if dataset_name == 'Powerplant_Airpollution':
        partialOrder = [['Installation_Type'],['Coal_Worker_Strike'],['Strike_Intervention'],['Strike_Resolution'],['Additional_Cost']]
        return partialOrder
    else:
        print(dataset_name)


def get_feature_labels(dataset_name):

    if dataset_name == 'Computer_Diagnostician':
       return  ['LBF', 'IBF', 'SS', 'RD', 'RO', 'RC']
    if dataset_name == 'Export_Textiles':
        return ['ES', 'ED', 'Pr']
    if dataset_name == 'Test_Strep':
        return ['TD', 'SI', 'TR', 'TRD', 'RH', 'Dfa', "Dws", 'Q']
    if dataset_name == 'LungCancer_Staging':
        return ['CT', 'MM', 'CTR', 'Ms', 'MsR', 'MsD', 'Tr', 'TD', 'LE']
    if dataset_name == 'HIV_Screening':
        return ['Sc', 'HS', 'HTR', 'TC', 'CMT', 'RRB', 'Q']
    if dataset_name == 'Powerplant_Airpollution':
        return ['IT', 'CWS', 'SI', 'SR', 'AC']





