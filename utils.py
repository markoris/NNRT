import dill
from torch.optim import Adam
def save(solver, fname="model_params.pt"):
    '''
    Saves a BundleIVP1D solver with the assumption that an Adam optimizer is used
    '''

    save_dict = solver.get_internals('all')
    save_dict['optimizer'] = solver.optimizer.state_dict()
    save_dict['type'] = solver.__class__
    with open("model_params.pt", 'wb') as fname:
        dill.dump(save_dict, fname)

    return

def load(fname):
    '''
    Loads a BundleIVP1D solver with the assumption of an Adam optimizer
    '''
    
    with open("model_params.pt", 'rb') as f:
        solver_params = dill.load(f)
    solver = solver_params["type"]( ode_system=solver_params['diff_eqs'], 
                                        conditions=solver_params['conditions'], 
                                        t_min=solver_params['r_min'][0],
                                        t_max=solver_params['r_max'][0],
                                        nets=solver_params['nets'],
                                        theta_min=solver_params['r_min'][1:],
                                        theta_max=solver_params['r_max'][1:],
                                        n_batches_valid=solver_params['n_batches'],
                                  )
    for key in solver_params.keys():
        if key in ["diff_eqs", "conditions", "r_min", "r_max", "n_batches", "global_epoch"]: 
            continue
        setattr(solver, key, solver_params[key])

    return solver

def train(solver, epochs=[500, 1000, 2000, 4000, 8000], lrs=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4]):

    for epoch, lr in zip(epochs, lrs):

        for g in solver.optimizer.param_groups:
            g['lr'] = lr
        #setattr(solver, 'optimizer', Adam(solver.nets[0].parameters(), lr=lr))
        solver.fit(max_epochs=epoch)
        #save(solver, fname='solver.chkp')

    return solver 
