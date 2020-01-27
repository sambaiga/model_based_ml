import torch
import pyro
import pyro.distributions as dist
import pyro.ops.stats as stats
import pandas as pd


def get_percentile_confidence_interval(samples, probs):
    pi = stats.pi(samples, prob=probs)
    below_interval_1 = (samples < pi[0].item()).sum().float() / len(samples)
    below_interval_2 = (samples < pi[1].item()).sum().float() / len(samples)
    
    df = {"LPI":[round(below_interval_1.item(), 2)*100, round(pi[0].item(),2)], 
          "UPI":[round(below_interval_2.item(), 2)*100, round(pi[1].item(),2)]}
    df = pd.DataFrame.from_dict(df)
    return df

def get_hpdi_confidence_interval(samples, probs):
    pi = stats.hpdi(samples, prob=probs)
    below_interval_1 = (samples < pi[0].item()).sum().float() / len(samples)
    below_interval_2 = (samples < pi[1].item()).sum().float() / len(samples)
    
    df = {"LPI":[round(below_interval_1.item(), 2)*100, round(pi[0].item(),2)], 
          "UPI":[100-round(below_interval_1.item(), 2)*100, round(pi[1].item(),2)]}
    df = pd.DataFrame.from_dict(df)
    return df


def get_chain_mode(samples, adj=0.01):
    
    silverman_factor = (0.75 * samples.size(0)) ** (-0.2)
    bandwidth = adj * silverman_factor * samples.std()
    x = torch.linspace(samples.min(), samples.max(), 1000)
    y = dist.Normal(samples, bandwidth).log_prob(x.unsqueeze(-1)).logsumexp(-1).exp()
    return x[y.argmax()]