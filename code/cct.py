import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

## Load the plant_knowledge data 
def load_data(filename = "data/plant_knowledge.csv"):
    """ 
    Loads csv file and returns, NumPy array
    without the Informant column
    """
    df = pd.read_csv(filename)
    data_matrix = df.drop(columns=["Informant"]).to_numpy()
    return data_matrix

def build_cct_model(X):
    """
    Cultural Consensus theory model 

    matrix shaoe (N_informants, M_items)

    returns model object 
    """
    N, M = X.shape

    with pm.Model() as model:
        # Competence of each informant : Uniform(0.5, 1.0)
        D = pm.Uniform("D", lower=0.5, upper=1.0, shape=N)

        #Consensus for each question: Bernoulli(0.5) prior
        Z = pm.Bernoulli("Z", p=0.5, shape = M)

        #reshape D t0 (N,1) -  across items 
        D_reshaped = D[:, None]

        #Compute pij using the CCT formula 
        p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped)

        #Observed responsed 
        X_obs = pm.Bernoulli("X_obs", p=p, observed=X)

    return model

def run_inference(model, draws=2000, tune=1000, chains=4, target_accept=0.9):
    """
    Model: pm.Model
    draws: int, samples to draw 
    tune: intm tuning steps 
    chains: int, number of MCMC chains
    target_accept: float, target acceptance rate for NUTS sampler 

    returns: arviz.InferenceData object 
    """
    with model:
        trace = pm.sample(draws=draws, tune= tune, chains=chains, target_accept= target_accept)
    return trace 

def analyze_results(trace):
    """
    visualize posterior samples of competence and consensus 

    trace: arviz.InferenceData
    """
    print("\nPosterior Summary:")
    summary = az.summary(trace, var_names=["D", "Z"], round_to=2)
    print(summary)

    #posterior for competence
    az.plot_posterior(trace, var_names=["D"], hdi_prob=0.94)
    plt.tight_layout()
    plt.show()

    #posterior consensus 
    az.plot_posterior(trace, var_names=["Z"], hdi_prob=0.94)
    plt.tight_layout()
    plt.show()

def compare_majority_vote(X, trace):
    """
    compare model consensus (Z) to simple majority vote per item 
    """
    majority_vote = (np.mean(X, axis= 0)> 0.5).astype(int)
    
    #model consensus 
    z_post_mean = trace.posterior["Z"].mean(dim=["chain", "draw"]).values

    #rounded model consensus
    model_consensus = (z_post_mean > 0.5).astype(int)

    print("Majoriity vote vs model consensus:")
    for j, (mv, mc, pm) in enumerate(zip(majority_vote, model_consensus, z_post_mean)):
        mark = "yes" if mv == mc else "no"
        print(f"Item {j:2d}: Majority = {mv}, Model = {mc} (p={pm:.2f}) {mark}")

if __name__ == "__main__":
    X = load_data()
    print("shape:", X.shape)
    print(X[:5])

    #test for model creation
    model = build_cct_model(X)
    print("model created")

    trace = run_inference(model)
    print("Sampling complete")

    analyze_results(trace) # the model converged 

    compare_majority_vote(X, trace)
