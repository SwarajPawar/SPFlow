

def get_decNode(dataset_name):

    if dataset_name == 'Dataset5':
        return ['Rework_Decision']
    elif dataset_name == 'Dataset1':
        return ['Export_Decision']
    elif dataset_name == 'Dataset2':
        return ['Treatment_Decision', 'Test_Decision']
    elif dataset_name == 'Dataset3':
        return ['Treatment', 'CT', 'Mediastinoscopy']
    elif dataset_name == 'Dataset4':
        return ['Screen', 'Treat_Counsel']
    elif dataset_name == 'Dataset6':
        return ['Strike_Intervention']
    else:
        print(dataset_name)


def get_utilityNode(dataset_name):

    if dataset_name == 'Dataset5':
        return ['Rework_Cost']
    if dataset_name == 'Dataset1':
        return ['Profit']
    if dataset_name == 'Dataset2':
        return ['QALE']
    if dataset_name == 'Dataset3':
        return ['Life_expectancy']
    if dataset_name == 'Dataset4':
        return ['QALE']
    if dataset_name == 'Dataset6':
        return ['Additional_Cost']

    else:
        print(dataset_name)

def get_scope_vars(dataset_name):

    #returns a list of all variables in sequence of partial order excluding decison variables
    #e.g.
    # if dataset_name == 'Dataset5':
    #     return ['System_State','Logic_board_fail', 'IO_board_fail', 'Rework_Outcome', 'Rework_Cost' ]

    partial_order = get_partial_order(dataset_name)
    decision_nodes = get_decNode(dataset_name)
    var_set = [var for var_set in partial_order for var in var_set]
    for d in decision_nodes:
        var_set.remove(d)
    return var_set



def get_partial_order(dataset_name):

    if dataset_name == 'Dataset5':
        partialOrder = [['System_State'], ['Rework_Decision'], ['Logic_board_fail', 'IO_board_fail', 'Rework_Outcome', 'Rework_Cost' ]]
        return partialOrder
    if dataset_name == 'Dataset1':
        partialOrder = [['Economical_State'], ['Export_Decision'],['Profit']]
        return partialOrder
    if dataset_name == 'Dataset2':
        partialOrder = [['Test_Decision'],['Test_Result', 'Streptococcal_Infection'],['Treatment_Decision'],
                        ['Rheumatic_Heart_Disease', 'Die_from_Anaphylaxis', 'Days_with_sore_throat', 'QALE']]
        return partialOrder
    if dataset_name == 'Dataset3':
        partialOrder = [['CT'],['CTResult', 'Mediastinal_Metastases'],['Mediastinoscopy'],
                               ['Mediastinoscopy_Result', 'Mediastinoscopy_death'], ['Treatment'], ['Treatment_Death', 'Life_expectancy' ]]
        return partialOrder
    if dataset_name == 'Dataset4':
        partialOrder = [['Screen'], ['HIV_Test_Result', 'HIV_Status'],['Treat_Counsel'],
                        ['Compliance_Medical_Therapy',	'Reduce_Risky_Behavior', 'QALE']]
        return partialOrder
    if dataset_name == 'Dataset6':
        partialOrder = ['Installation_Type','Coal_Worker_Strike','Strike_Resolution'],['Strike_Intervention'],['Additional_Cost']
        return partialOrder
    else:
        print(dataset_name)


def get_feature_labels(dataset_name):

    if dataset_name == 'Dataset5':
       return  ['SS', 'LBF', 'IBF', 'RO', 'RC']
    if dataset_name == 'Dataset1':
        return ['ES', 'Pr']
    if dataset_name == 'Dataset2':
        return ['TR', 'SI', 'RH', 'Dfa', "Dws", 'Q']
    if dataset_name == 'Dataset3':
        return ['CTR', 'MM', 'MsR', 'MsD', 'TD', 'LE']
    if dataset_name == 'Dataset4':
        return ['HTR', 'HS', 'CMT', 'RRB', 'Q']
    if dataset_name == 'Dataset6':
        return ['IT', 'CWS', 'SR', 'AC']





