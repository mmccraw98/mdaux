import pandas as pd

def load_sim_dataframes(sim_dir):
    """Load the simulation dataframes.
    
    Args:
        sim_dir (str): path to the simulation directory
        
    Returns:
        dpm_df, vertex_df, config_df (tuple): tuple of dpm, vertex and config dataframes
    """
    dpm_df = pd.read_csv(f'{sim_dir}/dpm_log.csv')
    vertex_df = pd.read_csv(f'{sim_dir}/vertex_log.csv')
    config_df = pd.read_csv(f'{sim_dir}/config.csv')
    macro_df = pd.read_csv(f'{sim_dir}/macro_log.csv')
    return dpm_df, vertex_df, config_df, macro_df


def getVertexData(df, v_id, dpm_id):
    """get the data for a specific vertex by its id and dpm id

    Args:
        df (dataframe): df containing the vertex data
        v_id (int): index of the vertex
        dpm_id (int): index of the dpm

    Returns:
        dataframe: subset of the df containing the vertex data
    """
    return df[(df.id == v_id) & (df.dpm_id == dpm_id)].copy()

def getParticleDataAsVector(df_subset_fixed_time, p_id, data_to_get='pos'):
    """get the data for a specific particle by its id
    
    Args:
        df_subset_fixed_time (dataframe): df containing the vertex data at a specific time
        p_id (int): index of the particle
        data_to_get (str): data to get (default: 'pos')
        
    Returns:
        list: list of data
    """
    if data_to_get == 'pos':
        return df_subset_fixed_time[df_subset_fixed_time.id == p_id][['x', 'y']].values[0]
    elif data_to_get == 'vel':
        return df_subset_fixed_time[df_subset_fixed_time.id == p_id][['vx', 'vy']].values[0]
    elif data_to_get == 'force':
        return df_subset_fixed_time[df_subset_fixed_time.id == p_id][['fx', 'fy']].values[0]
    else:
        raise ValueError('data_to_get must be pos, vel or force')

def getDpmData(df, dpm_id):
    """get the data for a specific dpm by its id

    Args:
        df (dataframe): df containing the vertex data
        dpm_id (int): index of the dpm

    Returns:
        dataframe: subset of the df containing the vertex data
    """
    return df[df.dpm_id == dpm_id].copy()

def getAtTime(df, time):
    """get the data for a specific time

    Args:
        df (dataframe): df containing the vertex data
        time (int): time

    Returns:
        dataframe: subset of the df containing the vertex data
    """
    return df[df.t == time].copy()

def getAtStep(df, step):
    """get the data for a specific step

    Args:
        df (dataframe): df containing the vertex data
        step (int): step

    Returns:
        dataframe: subset of the df containing the vertex data
    """
    return df[df.step == step].copy()