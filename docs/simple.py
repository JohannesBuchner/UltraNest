import scipy.stats

paramnames = ['param1', 'param2', 'param3']
centers = [0.4, 0.5, 0.6]
sigma = 0.1

def transform(cube):
    return cube

def loglike(theta):
    return scipy.stats.norm(centers, sigma).logpdf(theta).sum()

from ultranest import ReactiveNestedSampler
sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, 
    log_dir='my_gauss', # folder where to store files
    resume=True, # whether to resume from there (otherwise start from scratch) 
)

sampler.run(
    min_num_live_points=400,
    dlogz=0.5, # desired accuracy on logz
    min_ess=400, # number of effective samples
    update_interval_volume_fraction=0.4, # how often to update region
    max_num_improvement_loops=3, # how many times to go back and improve
)

sampler.print_results()

sampler.plot()
sampler.plot_trace()
