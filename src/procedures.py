import numpy as np
import functools as ft
from multiprocessing import Pool
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt
import src.utils as f
from src.epistemic_structures import modal_model


class Fitter(object):
    '''
    Take in a model instance and fit it to the data that the model instance carries
    '''
    def __init__(self, model, **kwargs):
        '''
        Optional arguments:
        nStart: number of starting point to use for the optimizer
        tol: tolerance level of optimizer convergence
        '''
        self.model = model
        self.num_subj = model.num_subj
        self.sample_size = model.num_phase 
        self.npz_directory = 'fitted_parameters/ModFit_' # where to save the fitted parameters
        self.convplt_directory = 'convergence_plot' # where to save the convergence plot of optimizers

        self.nStart = kwargs.pop('nStart', 20)
        self.tol = kwargs.pop('tol', 1e-12)
    def optimize(self, fname, bounds, niter, Data): # credit to the CCN Lab at Berkeley
        '''
         # bestparameters,bestllh = optimize(fname,bounds, Data,niter) runs
         the minimize function niter times on the function fname, with constraints
         bounds to find parameters that best fit the data Data. It returns the
         best likelihood and best fit parameters over the niter iterations
         # fname is the python function to optimize. fname should take as 
         first argument a 1 by n vector of parameters
         Note the bounds are set up differently than they are in Matlab,
         and should come as a list of [min,max] pairs. (ie. [[min,max],[min,max], ...])
         # Data is the data set to be fit by likelihood function fname
         # niter is the number of starting points for the optimization
         # best parameters is the 1*n vector of parameters found to minimize the
         negative log likelihood over the data
         # bestllh is the log likelihood value for the best parameters
         # optimcurve is the bestllh for each i in range(niter)
        '''
        outcomes = np.full([niter, len(bounds) + 1], np.nan)
        optimcurve = np.full(niter, np.nan)
        for i in range(niter):
            # random starting point based on maximum bounds
            params0 = self.model.initialize()
            # run the optimizer with constraints
            result = minimize(fun=fname, x0=params0, args=(Data), bounds=bounds, tol = self.tol)
            x, bestllh = result.x, result.fun
            outcomes[i, :] = [bestllh] + [xi for xi in x]
            optimcurve[i] = min(outcomes[:(i + 1), 0])
        # find the global minimum out of all outcomes
        i = np.argwhere(outcomes[:, 0] == np.min(outcomes[:, 0]))
        bestparameters = outcomes[i[0], 1:].flatten()
        bestllh = outcomes[i[0], 0].flatten()[0]
        return (bestparameters, bestllh, optimcurve)
    def subj_fit(self, Data):
        '''
        Fit one subject
        Data (numpy matrix): subject data
        '''
        bounds, niter = self.model.parameter_space, self.nStart # set up parameter bound and sample size
        if not self.model.discrete_parameter: # if does not have discrete parameter
            bestparam, bestllh, optimcurve = self.optimize(self.model.nLLH, bounds, niter, Data) # fit
        else:
            discreteParams, subBestparams, subBestllhs, suboptimcurve = self.model.parameter_combination, [], [], []
            for combo in discreteParams: # for each combination of discrete parameter value
                print(combo)
                subBestparam, subBestllh, suboptimcurve = self.optimize(ft.partial(self.model.nLLH, discrete_param = combo),bounds, niter, Data)  # fit
                subBestparams.append(subBestparam)
                subBestllhs.append(subBestllh)
            argmin = np.argmin(subBestllhs)
            bestparam, bestllh, optimcurve = np.array(list(discreteParams[argmin]) + list(subBestparams[argmin])), subBestllhs[argmin], suboptimcurve[argmin]
            # don't forget the convention that discrete precedes continuous
        aic, bic = f.get_AIC(bestllh, self.numParam), f.get_BIC(bestllh, self.numParam, self.sample_size)
        return aic, bic, bestparam, optimcurve
    def save_fit(self, sub, verbose=True):
        '''
        Save the fitting result of one subject
        '''
        nSub = self.num_subj # number of subjects
        if verbose:
            print('subject: ' + str(sub))
        aic, bic, param, _ = self.subj_fit(self.model.data[sub-1]) # fit the currect subject
        try: # see if there is an existing saved file
            ModFit = np.load(self.npz_directory + self.model.name + '.npz', allow_pickle=True) # if no error returned
        except: # if no existing file, np.load would return error and so we initialize a file.
            AICs, BICs, niters, Params = np.empty(nSub), np.empty(nSub), \
                                             np.empty(nSub), np.empty(nSub,dtype=list)
        else: # get existing file of fitted values
            AICs, BICs, niters, Params = ModFit['AICs'], ModFit['BICs'], ModFit['niters'], \
                                             ModFit['Params']  # get existing fitted values
        niters[sub - 1] = self.nStart
        AICs[sub - 1], BICs[sub - 1], Params[sub - 1] = aic, bic, param
        np.savez(self.npz_directory + self.model.name, AICs=AICs, BICs=BICs, Params=Params, niters=niters,
                 agtName=self.model.name)  # save the file including the current subject data and replace the old one
    def parallel_fit(self, participants, nCore=1):
        '''
        Note: It might not work on IDEs. Run on terminal directly.
        Args:
            participants: np array of subject numbers
            nCore: cores you want to use on your computer
        '''
        print('now fitting ' + self.model.name + ' make sure your computer is Caffeinated')
        start_time = time.time()  # register the start time
        pool = Pool(nCore) # number of processing core to use
        pool.map(ft.partial(self.save_fit), participants)
        print("--- %s seconds ---" % (time.time() - start_time))
    def fillin_fit(self):
        """
        Sometimes parallel fitting will omit some subjects
        This function checks where fitted parameter is absent and refit to fill it in
        """
        ModFit = np.load(self.npz_directory + self.model.name + '.npz', allow_pickle=True)
        AICs, BICs, niters, Params = ModFit['AICs'], ModFit['BICs'], ModFit['niters'], \
                                             ModFit['Params'] # get existing fitted values
        num = len(AICs)
        for sub in range(num):
            if Params[sub] is None:
                print('filling in subj '+str(sub+1))
                AICs[sub], BICs[sub], Params[sub], _ = self.subj_fit(self.model.data[sub])
                niters[sub] = self.nStart
        np.savez(self.npz_directory + self.model.name, AICs=AICs, BICs=BICs, Params=Params, niters=niters,
                 agtName=self.model.name)
    def plot_convergence(self, agent_name, niter, optimcurve):
        '''
        Convergence plot to see whether nStart is large enough to find global minimum
        Args:
            agent_name: name of the model
            niter: number of starting point
            optimcurve: the minimum value of each starting point
        '''
        assert len(niter) == len(optimcurve)
        plt.figure()
        for sub in range(len(niter)):
            opt, it = optimcurve[sub], niter[sub]
            if opt is None or it is None: # if None due to computer error
                continue
            plt.plot(range(int(it)), np.round(opt, 6), 'o-')
        plt.xlabel('iteration')
        plt.ylabel('best minimum')
        plt.title('convergence plot for '+ agent_name)
        plt.savefig(self.convplt_directory+'/convplt_'+ agent_name)
        plt.show()

class Simulator(object):
    '''
    Take in a model instance and simulate data
    '''
    def __init__(self, model, nSim=100):
        self.model = model
        self.nSim = nSim
        self.modal_model = modal_model() # help compute some normative information about the game
        self.npz_directory = 'fitted_parameters/ModFit_' # where to import the fitted parameters
        self.val_directory = 'validation_data/' # where to save the simulated data for model validation
        # Important: discrete go before continuous parameter

    def add_columns(self, data):
        '''
        Compute and append some extra columns to the simulated data
        Takes time if data is large
        '''
        print("now adding columns: should_know and numRound")
        should_know, numRound = [], []
        for row in range(data.shape[0]):
            order, cards = data['order'].iloc[row], data['cards'].iloc[row]
            order = order.split(",")
            temp = self.modal_model.compute_subj_response(cards, order)
            numRound.append(len(eval(data['outcomeArray'].iloc[row])))
            if not temp[-1][-1]: # if subject eventually shouldn't know
                should_know.append(10)
            else:
                should_know.append(f.getSizeOfNestedList(temp))
        data['should_know'], data['numRound'] = should_know, numRound

        print("now adding columns: cards_iso and inference_level")
        iso_map = self.modal_model.iso_map
        cards_iso, inference_level = [], []
        for row in range(data.shape[0]):
            order_str = data['order'].iloc[row]
            order, cards = order_str.split(","), data['cards'].iloc[row]
            cards_iso.append(iso_map[cards])
            inference_level.append(self.modal_model.inference_level[iso_map[cards] + ':' + order_str])
        data['cards_iso'], data['inference_level'] = cards_iso, inference_level
        return data
    def simulation(self, Param, AICs, BICs):
        '''
        This function is to be called by other functions to simulate data for one set of parameters
        Param: The simulating parameters of all subjects (list of lists)
        AICs, BICs: The AIC and BIC of the parameters of all subjects just to attach to the output data (list)
        return: a pandas dataframe
        '''
        print('now simulating ' + self.model.name)
        colnamesOut = self.model.colnamesOut + [name for name in self.model.parameter_names] + ['aic', 'bic', 'iSim'] # add parameter names to the data
        Data = np.ones(len(colnamesOut))  # initialize output learning data
        for n in range(self.model.num_subj): # for every subject
            print('subject ' + str(n+1))
            p, aic, bic = Param[n], AICs[n], BICs[n]
            for s in range(self.nSim): # for every simulation
                sub = self.model.agent(n+1, p) # simulate agent data using corresponding param
                sub = np.insert(sub, sub.shape[1], np.repeat(aic, sub.shape[0]), axis=1) # add also the aic
                sub = np.insert(sub, sub.shape[1], np.repeat(bic, sub.shape[0]), axis=1) # add also the bic
                sub = np.insert(sub, sub.shape[1], np.repeat(s+1, sub.shape[0]), axis=1) # add also the simulation index
                Data = np.vstack((Data, sub))
        return f.np2pd(Data[1:], colnamesOut)
    def validation_simulation(self):
        '''
        Simulate data using fitted parameter values
        return: None but saves the simulated data to csv
        '''
        ModFit = np.load(self.npz_directory + self.model.name + '.npz', allow_pickle=True)
        Param, AICs, BICs = ModFit['Params'], ModFit['AICs'], ModFit['BICs']
        data = self.add_columns(self.simulation(Param, AICs, BICs)) #takes some time to add the columns if nSim is large
        data.to_csv(r'validation_data/' + self.model.name + '.csv', index=False, header=True)
    
    