import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import random

def generate_fitting_dates(data : pd.DataFrame, date_method: str, rollyears=20) :
    """
    generate a list 4 tuples, one element for each year in the data
    each tuple contains [fit_start, fit_end, period_start, period_end] datetime objects
    the last period will be a 'stub' if we haven't got an exact number of years
    
    date_method can be one of 'in_sample', 'expanding', 'rolling'
    
    if 'rolling' then use rollyears variable 
    
    note that:
    fit_tuple[0] is the fit start date
    fit_tuple[1] is the fit end date
    fit_tuple[2] is the period start date
    fit_tuple[3] is the period end date 
    """
    
    start_date: datetime=data.index[0]
    end_date:datetime=data.index[-1]
    
    ## generate list of dates, one year apart, including the final date
    yearstarts=list(pd.date_range(start_date, end_date, freq="12ME"))+[end_date]
   
    ## loop through each period     
    periods=[]
    for tidx in range(len(yearstarts))[1:-1]:
        ## these are the dates we test in
        period_start=yearstarts[tidx]
        period_end=yearstarts[tidx+1]

        ## now generate the dates we use to fit
        if date_method=="in_sample":
            fit_start=start_date
        elif date_method=="expanding":
            fit_start=start_date
        elif date_method=="rolling":
            yearidx_to_use=max(0, tidx-rollyears)
            fit_start=yearstarts[yearidx_to_use]
        else:
            raise Exception("don't recognise date_method %s" % date_method)
            
        if date_method=="in_sample":
            fit_end=end_date
        elif date_method in ['rolling', 'expanding']:
            fit_end=period_start
        else:
            raise Exception("don't recognise date_method %s " % date_method)
        
        periods.append([fit_start, fit_end, period_start, period_end])

    ## give the user back the list of periods
    return periods

def markosolver(returns: pd.DataFrame, equalisemeans=False, equalisevols=True, default_vol=0.2, default_SR=1.0):
    """
    Returns the optimal portfolio for the dataframe returns
    
    If equalisemeans=True then assumes all assets have same return if False uses the asset means    
    
    If equalisevols=True then normalises returns to have same standard deviation; the weights returned
       will be 'risk weightings'
       
    Note if usemeans=True and equalisevols=True effectively assumes all assets have same sharpe ratio
    
    """    

    if equalisevols:
        use_returns=_equalise_vols(returns, default_vol)
    else:
        use_returns=returns

    ## Sigma matrix: Calculates the covariance matrix of
    sigma=use_returns.cov().values

    ## Expected mean returns    
    if equalisemeans:
        ## Don't use the data - Set to the average Sharpe Ratio
        avg_return=np.mean(use_returns.mean())
        mus=np.array([avg_return for asset_name in use_returns.columns], ndmin=2).transpose()

    else:
        # get mean returns of each rule/asset
        mus=np.array([use_returns[asset_name].mean() for asset_name in use_returns.columns], ndmin=2).transpose()
    
    ## Starting weights
    number_assets=use_returns.shape[1]  #number of assets / rules
    start_weights=[1.0/number_assets]*number_assets # starting will equal weights
    
    ## Constraints - positive weights, adding to 1.0
    bounds=[(0.0,1.0)]*number_assets #weights must be between 0 and 1
    cdict=[{'type':'eq', 'fun':_addem}] #sum of weights on each row must be equal to 1
    
    # we actually want to maximize SR, but algo only produce minimization
    # so we make sr negative and try to minimize it
    ans=minimize(_neg_SR, start_weights, (sigma, mus), method='SLSQP', bounds=bounds, constraints=cdict, tol=0.00001)
        
    return ans['x']

def bootstrap_portfolio(returns_to_bs: pd.DataFrame, monte_carlo=200, monte_length=250, equalisemeans=False, equalisevols=True, default_vol=0.2, default_SR=1.0):
    """
    Given dataframe of returns; returns_to_bs, performs a bootstrap optimisation
    
    We run monte_carlo numbers of bootstraps
    Each one contains monte_length days drawn randomly, with replacement 
    (so *not* block bootstrapping)
    
    The other arguments are passed to the optimisation function markosolver
    
    Note - doesn't deal gracefully with missing data. Will end up downweighting stuff depending on how
      much data is missing in each boostrap. You'll need to think about how to solve this problem. 
    
    
    
    IMPORTANT NOTE:
    we see that bs_idx give us random date that is not in order, eg it can be 2024/10/10, 2020/01/01, 2020/06/15, ...
    we don't care about the order, because the 'minimize' function from 'markosolver'
    will find the best weights such that we will get the 'daily' highest Sharpe Ratio
    since it is daily, we don't care about the order of the date
    
    """
    weightlist=[]
    for _ in range(monte_carlo):
        # pick an index randomly from the dataframe
        bs_idx=[int(random.uniform(0,1)*len(returns_to_bs)) for i in range(monte_length)]
        # slice the dataframe to be between the random index and the last index
        returns=returns_to_bs.iloc[bs_idx,:] 
        weight=markosolver(returns, equalisemeans=equalisemeans, equalisevols=equalisevols, default_vol=default_vol, default_SR=default_SR)
        weightlist.append(weight)
    ### We can take an average here; only because our weights always add up to 1. If that isn't true
    ###    then you will need to some kind of renormalisation
     
    theweights_mean=list(np.mean(weightlist, axis=0)) # get the mean of weight of each rule/asset (average of each column)
    return theweights_mean

def optimise_over_periods(data: pd.DataFrame, date_method: str, fit_method: str, rollyears=20, equalisemeans=False, equalisevols=True, 
                          monte_carlo=200, monte_length=250, shrinkage_factors=(0.5, 0.5)):
    """
    Do an optimisation
    
    Returns data frame of weights
    
    Note if fitting in sample weights will be somewhat boring
    
    Doesn't deal with eg missing data in certain subperiods
    
    
    """
    weekly_data=data.resample("W").sum()
    
    ## Get the periods
    fit_periods=generate_fitting_dates(weekly_data,date_method = date_method, rollyears=rollyears)
    
    ## Do the fitting
    ## Build up a list of weights, which we'll concat
    weight_list=[]
    for fit_tuple in fit_periods:
        ## Fit on the slice defined by first two parts of the tuple
        period_subset_data=weekly_data[fit_tuple[0]:fit_tuple[1]]
        
        ## Can be slow, if bootstrapping, so indicate where we are
        
        # print "Fitting data for %s to %s" % (str(fit_tuple[2]), str(fit_tuple[3]))
        print("Fitting data for %s to %s" % (str(fit_tuple[2]), str(fit_tuple[3])))
        print(f"With data from {str(fit_tuple[0])} to {str(fit_tuple[1])}")
        print("-"*40)
        period_subset_data = _apply_exponential_weight(period_subset_data, halflife=52*5) #apply weighted to attenuate data, while the last five years are more important
        
        if fit_method=="one_period":
            weights=markosolver(period_subset_data, equalisemeans=equalisemeans, equalisevols=equalisevols)
        elif fit_method=="bootstrap":
            weights=bootstrap_portfolio(period_subset_data, equalisemeans=equalisemeans, 
                                        equalisevols=equalisevols, monte_carlo=monte_carlo, 
                                        monte_length=monte_length)                    
        else:
            raise Exception("Fitting method %s unknown" % fit_method)
        
        ## We adjust dates slightly to ensure no overlaps
        dindex=[fit_tuple[2]+datetime.timedelta(seconds=1), fit_tuple[3]-datetime.timedelta(seconds=1)]
        
        ## create a double row to delineate start and end of test period
        weight_row=pd.DataFrame([weights]*2, index=dindex, columns=weekly_data.columns)
        
        weight_list.append(weight_row)
        
    weight_df=pd.concat(weight_list, axis=0)
    
    return weight_df

def _equalise_vols(returns: pd.DataFrame, default_vol):
    """
    Normalises returns so they have the in sample vol of defaul_vol (annualised)
    Assumes daily returns
    """
    
    factors=(default_vol/np.sqrt(252))/returns.std(axis=0) #divide by sqrt(252) to get daily vol, then get factor for normalisation
    facmat=create_dull_pd_matrix(dullvalue=factors, dullname=returns.columns, index=returns.index) #create matrix of factors
    norm_returns=returns*facmat.values
    norm_returns.columns=returns.columns

    return norm_returns

def _addem(weights):
    ## Used for constraints
    return 1.0 - sum(weights)

def create_dull_pd_matrix(dullvalue=0.0, dullname="A", startdate=pd.Timestamp(1970,1,1).date(), enddate=datetime.datetime.now().date(), index=None):
    """
    create df where columns share the same value on all dates
    e.g. if we have dullvalue of [0, 1, 2], index of [2020-01-01, 2020-01-02, 2020-01-03]
    we get df of:
    
    time       A    B    C
    2020-01-01 0    1    2
    2020-01-02 0    1    2
    2020-01-03 0    1    2
    """
    
    if index is None:
        index=pd.date_range(startdate, enddate)
    
    dullvalue=np.array([dullvalue]*len(index))
    
    ans=pd.DataFrame(dullvalue, index, columns=[dullname])
    
    return ans

def _neg_SR(weights, sigma, mus):
    ## Returns minus the Sharpe Ratio (as we're minimising)

    """    
    estreturn=250.0*((np.matrix(x)*mus)[0,0])
    variance=(variance(x,sigma)**.5)*16.0
    """
    estreturn=(np.matrix(weights)*mus)[0,0]
    std_dev=(_variance(weights,sigma)**.5)

    
    #share_ratio = return / risk
    return -estreturn/std_dev

def _variance(weights, sigma):
    ## returns the variance (NOT standard deviation) given weights and sigma
    return (np.matrix(weights)*sigma*np.matrix(weights).transpose())[0,0]

def _apply_exponential_weight(df, halflife=None, alpha=None, columns=None):
    """
    Apply exponential weighting to specified columns of a DataFrame.
    
    Parameters:
    - df: Input DataFrame
    - halflife: Number of periods for weight to reduce by half
    - alpha: Explicit decay rate (alternative to halflife)
    - columns: List of columns to weight (if None, weight all columns)
    
    Returns:
    - DataFrame with weighted columns
    """
    # Calculate alpha from halflife if not provided
    if halflife is not None:
        alpha = 1 - np.exp(np.log(0.5) / halflife)
    elif alpha is None:
        # Default to a standard decay if neither halflife nor alpha is provided
        alpha = 0.3
    
    # Validate alpha
    if not 0 < alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    # Calculate weights
    weights = np.power(1 - alpha, np.arange(len(df))[::-1])
    
    # Create a copy of the DataFrame to avoid modifying the original
    weighted_df = df.copy()
    
    # Determine which columns to weight
    if columns is None:
        columns = df.columns
    
    # Multiply specified columns by the weights
    for column in columns:
        weighted_df[column] = df[column] * weights
    
    return weighted_df