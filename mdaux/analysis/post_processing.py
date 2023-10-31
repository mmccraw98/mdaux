import numpy as np
from joblib import Parallel, delayed
from .data_manipulation import getAtStep, getDpmData
from mdaux.utils.helpers import compute_axes_of_polygon
from tqdm import tqdm

def unwrap_major_minor_axis_angles(theta_max, theta_min, tol=0.1):
    """Unwraps the major and minor axis angles to be continuous.
    
    Parameters
    ----------
        theta_max : array-like
        The major axis angles.
        theta_min : array-like
        The minor axis angles.
        tol : float
        The tolerance for detecting a boundary crossing.
        
    Returns
    -------
        theta_max : array-like
        The unwrapped major axis angles.
        theta_min : array-like
        The unwrapped minor axis angles.
    """

    # theta_max and theta_min will ALWAYS be separated by +- pi / 2
    # thus, only one of these may cross the boundary at a given instant (there should really be some delay between crossings)
    # also, the noise SHOULD be constant between the two signals, so we can just take the difference between the two to cancel out the noise
    # which gives us the crossing points
    # we then need to use the crossing points to unwrap the boundary crossings

    diff_shifts = abs(np.diff(theta_max - theta_min))
    theta_max_diffs = - np.diff(theta_max) * diff_shifts
    theta_min_diffs = - np.diff(theta_min) * diff_shifts

    max_shifts = np.zeros(theta_max.size)
    min_shifts = np.zeros(theta_min.size)
    for i in range(1, theta_max.size - 1):
        if abs(theta_max_diffs[i]) > tol or abs(theta_min_diffs[i]) > tol:
            if abs(theta_max_diffs[i]) > abs(theta_min_diffs[i]):
                max_shifts[i + 1:] += np.pi * np.sign(theta_max_diffs[i])
            else:
                min_shifts[i + 1:] += np.pi * np.sign(theta_min_diffs[i])
    return theta_max + max_shifts, theta_min + min_shifts

def autocorrFFT(x):
    N = len(x)
    F = np.fft.fft(x, n=2 * N)  # 2 * N because of zero-padding
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res = (res[:N]).real   # now we have the autocorrelation in convention B
    n = N * np.ones(N) - np.arange(0, N) #d ivide res(m) by (N-m)
    return res / n # this is the autocorrelation in convention A

def msd_fft(r, t):
    # adapted from: https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft
    N = len(r)
    D = np.square(r).sum(axis=1)
    D = np.append(D, 0)
    S2 = sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    Q = 2 * D.sum()

    S1 = np.zeros(N)
    for m in range(N):
        Q = Q - D[m - 1] - D[N - m]
        S1[m] = Q / (N - m)
    return (S1 - 2 * S2)[1:], t[1:]

def simulateNonInteractingRandomWalk(N_steps, N_particles=1, N_dim=2, step_size=1, dt=1):
    disps = step_size * (1 - 2 * np.round(np.random.rand(N_steps, N_dim, N_particles)))  # important step to ROUND the random numbers so that the steps are +step or -step!
    return np.cumsum(disps, axis=0), np.linspace(0, N_steps, N_steps) * dt

def fit_msd_powerlaw(msd, tau, tau_min=None, tau_max=None):
    if tau_min is None:
        tau_min = min(tau[tau > 0])
    if tau_max is None:
        tau_max = max(tau / 10)
    else:
        tau_max = tau_max
    mask = (tau > tau_min) & (tau < tau_max)
    params = np.polyfit(np.log10(tau[mask]), np.log10(msd[mask]), 1)
    return params[0], 10 ** params[1]

def get_diffusion_coef_and_exponent(msd, tau, n_dim, tau_min=None, tau_max=None):
    # msd = 2 * n_dim * D * tau (for a random walk)
    # we fit the msd to a power law of the form msd = A * tau ^ alpha
    exponent, coef = fit_msd_powerlaw(msd, tau, tau_min=tau_min, tau_max=tau_max)
    return coef / (2 * n_dim), exponent, coef  # last is just for convenience

def msd_ensemble(trajectories, time):
    _, tau = msd_fft(trajectories[0], time)
    return np.mean([msd_fft(traj, time)[0] for traj in trajectories if not np.any(np.isnan(traj))], axis=0), tau


def process_major_minor_axes_for_step(dpm_df, vertex_df, step):
    results = []
    dpm_df_step = getAtStep(dpm_df, step)
    vertex_df_step = getAtStep(vertex_df, step)
    
    for dpm_id in dpm_df_step.dpm_id.unique():
        vertices = getDpmData(vertex_df_step, dpm_id)
        points = vertices[['x', 'y']].values
        try:
            major_axis_length, minor_axis_length, major_axis_vector, minor_axis_vector, major_axis_theta, minor_axis_theta = compute_axes_of_polygon(points)

            results.append((step, dpm_id, [
                major_axis_length, minor_axis_length,
                major_axis_vector[0, 0], major_axis_vector[0, 1],
                major_axis_vector[1, 0], major_axis_vector[1, 1],
                minor_axis_vector[0, 0], minor_axis_vector[0, 1],
                minor_axis_vector[1, 0], minor_axis_vector[1, 1],
                major_axis_theta, minor_axis_theta
            ]))
        except:
            pass  # TODO: Handle this better

    return results

def append_major_minor_axes_to_dpmdf(dpm_df, vertex_df):
    columns_to_init = [
        'major_axis', 'minor_axis',
        'major_axis_x_i', 'major_axis_y_i',
        'major_axis_x_f', 'major_axis_y_f',
        'minor_axis_x_i', 'minor_axis_y_i',
        'minor_axis_x_f', 'minor_axis_y_f',
        'major_axis_theta', 'minor_axis_theta'
    ]

    dpm_df[columns_to_init] = 0

    all_results = Parallel(n_jobs=-1)(delayed(process_major_minor_axes_for_step)(dpm_df, vertex_df, step) for step in tqdm(dpm_df.step.unique(), desc='Processing dataframes'))

    # Flatten the list of results
    all_results = [item for sublist in all_results for item in sublist]

    # Update the DataFrame
    for step, dpm_id, values in tqdm(all_results, desc='Updating DataFrame'):
        dpm_df.loc[(dpm_df.step == step) & (dpm_df.dpm_id == dpm_id), columns_to_init] = values