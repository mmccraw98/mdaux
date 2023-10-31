from mdaux.analysis.post_processing import simulateNonInteractingRandomWalk, msd_ensemble, get_diffusion_coef_and_exponent
from tqdm import tqdm

def test_random_walk_diffusion_coef(n_dim, N_steps, dt=1, step_size=1):  # TODO maybe move this to a separate file (and definitely make more tests for other functions!)
  traj, time = simulateNonInteractingRandomWalk(N_steps, N_particles=100, N_dim=n_dim, dt=dt, step_size=step_size)
  msd, tau = msd_ensemble([traj[:, :, i] for i in tqdm(range(traj.shape[-1]))], time)
  D, alpha, _ = get_diffusion_coef_and_exponent(msd, tau, n_dim)
  D_theory = n_dim * step_size ** 2 / (2 * dt * n_dim)
  print('D = %f, D_theory = %f' % (D, D_theory))  # TODO change to assertion
  print('alpha = %f' % alpha)