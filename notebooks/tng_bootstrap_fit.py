# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: py311-main
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Summary:
#
# In this notebook, I just load the TNG data and repeatedly fit the data with an increasing number of randonly drawn satellites above a certain mass threshold. Question: How many TNG satellites are needed to demonstrate that a sinsusoid function is necessary.
#
# In order to calculate this, I increase the number of satellites in the sample by a set amount (not too finely spaced, otherwise the code will take too long) and for that number of satellites, calculate the MCMC fit 1,000-10,000 times in order to get a distribution on the b paramater mean divided by the b parameter standard deviation. 

# %%
# import libraries, set font, ect.

import matplotlib.pyplot as plt 
# %matplotlib inline 

import astropy as ap
from astropy import units as u
from fractions import Fraction 
import astroquery as aq
import numpy as np
from astropy.table import Table
import array as arr
import pandas as pd
import scipy.stats
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astropy.table import Table, join

import scipy.interpolate as interp
import scipy
import emcee
import corner
import sys

import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], 
    "mathtext.fontset": "cm",  
    "axes.unicode_minus": False
})


# %% [markdown]
# ## MCMC fit: 

# %%
def boostrap_90(data,sf_index,q_index,N):
    
    bins = 18 # same as boostrap function
    dphi = 90/bins/2
    n_bootstrap = N  # number of bootstrap resamples

    # arrays to hold bootstrapped quenched fractions
    boot_fq = np.zeros((n_bootstrap, bins))

    # loop over bootstrap resamples
    for i in range(n_bootstrap):

        sf_sample = data.loc[sf_index].sample(frac=1, replace=True)
        q_sample  = data.loc[q_index].sample(frac=1, replace=True)
        
        g_N_sf, _ = np.histogram(sf_sample['alpha'],density=False, range=[0,90], bins=bins)
        g_N_q, _  = np.histogram(q_sample['alpha'],density=False, range=[0,90], bins=bins)

        with np.errstate(divide="ignore", invalid="ignore"):
            fq = g_N_q / (g_N_q + g_N_sf)
        fq[np.isnan(fq)] = 0.0  # handle empty bins
        
        boot_fq[i] = fq
    
    fq_mean = np.nanmean(boot_fq, axis=0)
    fq_std = np.nanstd(boot_fq, axis=0)

    return fq_mean, fq_std



# %%
# load TNG100 data
# just once 

angle_array = np.loadtxt('angle_array.txt')

# load 1e8 data

df_tng100 = pd.read_csv("satellite_1e8/centrals_satellites_tng100_mstar_1e8.csv")
df_tng100_host_mh = pd.read_csv("satellite_1e8/centrals_satellites_tng100_host_mh_1e8.csv")
df_tng100_alpha = pd.read_csv("satellite_1e8/centrals_satellites_tng100_alpha_1e8.csv")
tng100_sfr_info = pd.read_csv("satellite_1e8/tng100_sfr_info_1e8.csv")

index = (df_tng100_host_mh['host_mh'] > 12) & (df_tng100_host_mh['host_mh'] < 12.5)
sfr_interp = np.log10((10**(0.75*tng100_sfr_info['mstar'][index]-7.5))/10)
index_sf = (10**sfr_interp <= tng100_sfr_info['sfr'][index])
index_q = (10**sfr_interp > tng100_sfr_info['sfr'][index])
q_sf_array = np.ones(len(tng100_sfr_info['host_mh'][index]))
q_sf_array[index_sf] = 0.

# load 1e7 data

df_tng100 = pd.read_csv("satellite_1e7/centrals_satellites_tng100_mstar_1e7.csv")
df_tng100_host_mh = pd.read_csv("satellite_1e7/centrals_satellites_tng100_host_mh_1e7.csv")
df_tng100_alpha = pd.read_csv("satellite_1e7/centrals_satellites_tng100_alpha_1e7.csv")
tng100_sfr_info = pd.read_csv("satellite_1e7/tng100_sfr_info_1e7.csv")

index = (df_tng100_host_mh['host_mh'] > 12) & (df_tng100_host_mh['host_mh'] < 12.5)
sfr_interp = np.log10((10**(0.75*tng100_sfr_info['mstar'][index]-7.5))/10)
index_sf = (10**sfr_interp <= tng100_sfr_info['sfr'][index])
index_q = (10**sfr_interp > tng100_sfr_info['sfr'][index])
q_sf_array = np.ones(len(tng100_sfr_info['host_mh'][index]))
q_sf_array[index_sf] = 0.

# load 1e6 data

df_tng100 = pd.read_csv("satellite_1e6/centrals_satellites_tng100_mstar_1e6.csv")
df_tng100_host_mh = pd.read_csv("satellite_1e6/centrals_satellites_tng100_host_mh_1e6.csv")
df_tng100_alpha = pd.read_csv("satellite_1e6/centrals_satellites_tng100_alpha_1e6.csv")
tng100_sfr_info = pd.read_csv("satellite_1e6/tng100_sfr_info_1e6.csv")

index = (df_tng100_host_mh['host_mh'] > 12) & (df_tng100_host_mh['host_mh'] < 12.5)
sfr_interp = np.log10((10**(0.75*tng100_sfr_info['mstar'][index]-7.5))/10)
index_sf = (10**sfr_interp <= tng100_sfr_info['sfr'][index])
index_q = (10**sfr_interp > tng100_sfr_info['sfr'][index])
q_sf_array = np.ones(len(tng100_sfr_info['host_mh'][index]))
q_sf_array[index_sf] = 0.

# %%
tng100_sfr_info[index]

# %%
subsample = tng100_sfr_info[index].sample(n=2,replace=True,random_state=rng.integers(0, 2**32))


# %%
def boostrap_90(data,sf_index,q_index,N):
    
    bins = 18 # same as boostrap function
    dphi = 90/bins/2
    n_bootstrap = N  # number of bootstrap resamples

    # arrays to hold bootstrapped quenched fractions
    boot_fq = np.zeros((n_bootstrap, bins))

    # loop over bootstrap resamples
    for i in range(n_bootstrap):

        sf_sample = data.loc[sf_index].sample(frac=1, replace=True)
        q_sample  = data.loc[q_index].sample(frac=1, replace=True)
        
        g_N_sf, _ = np.histogram(sf_sample['alpha'],density=False, range=[0,90], bins=bins)
        g_N_q, _  = np.histogram(q_sample['alpha'],density=False, range=[0,90], bins=bins)

        with np.errstate(divide="ignore", invalid="ignore"):
            fq = g_N_q / (g_N_q + g_N_sf)
        fq[np.isnan(fq)] = 0.0  # handle empty bins
        
        boot_fq[i] = fq
    
    fq_mean = np.nanmean(boot_fq, axis=0)
    fq_std = np.nanstd(boot_fq, axis=0)

    return fq_mean, fq_std

def bootstrap_90(data, sf_index, q_index, N, bins=18, seed=None):

    rng = np.random.default_rng(seed)
    boot_fq = np.zeros((N, bins))

    base = data.index

    for i in range(N):

        # sample indices ONCE
        sampled_idx = rng.choice(base, size=len(base), replace=True)

        d = data.loc[sampled_idx]
        sf = sf_index.loc[sampled_idx]
        q  = q_index.loc[sampled_idx]

        sf_sample = d.loc[sf]
        q_sample  = d.loc[q]

        g_N_sf, _ = np.histogram(sf_sample['alpha'], bins=bins, range=(0, 90))
        g_N_q,  _ = np.histogram(q_sample['alpha'],  bins=bins, range=(0, 90))

        with np.errstate(divide="ignore", invalid="ignore"):
            fq = g_N_q / (g_N_q + g_N_sf)

        fq[np.isnan(fq)] = 0.0
        boot_fq[i] = fq

    return np.nanmean(boot_fq, axis=0), np.nanstd(boot_fq, axis=0)

def bootstrap_90(data, sf_index, q_index, N, bins=18, seed=None):

    rng = np.random.default_rng(seed)
    boot_fq = np.zeros((N, bins))

    base_idx = data.index

    for i in range(N):

        # inner bootstrap (with replacement)
        sampled_idx = rng.choice(
            base_idx,
            size=len(base_idx),
            replace=True
        )

        d  = data.loc[sampled_idx]
        sf = sf_index.loc[sampled_idx]
        q  = q_index.loc[sampled_idx]

        sf_sample = d.loc[sf]
        q_sample  = d.loc[q]

        g_N_sf, _ = np.histogram(sf_sample['alpha'], bins=bins, range=(0, 90))
        g_N_q,  _ = np.histogram(q_sample['alpha'],  bins=bins, range=(0, 90))

        with np.errstate(divide="ignore", invalid="ignore"):
            fq = g_N_q / (g_N_q + g_N_sf)

        fq[np.isnan(fq)] = 0.0
        boot_fq[i] = fq

    return np.nanmean(boot_fq, axis=0), np.nanstd(boot_fq, axis=0)


# %%
def bootstrap_90(data, sf_index, q_index, N, bins=18, seed=None):

    rng = np.random.default_rng(seed)
    boot_fq = np.zeros((N, bins))

    base_idx = data.index

    for i in range(N):

        # inner bootstrap (with replacement)
        sampled_idx = rng.choice(base_idx,=len(base_idx),replace=True)

        d  = data.loc[sampled_idx]
        sf = sf_index.loc[sampled_idx]
        q  = q_index.loc[sampled_idx]

        sf_sample = d.loc[sf]
        q_sample  = d.loc[q]

        g_N_sf, _ = np.histogram(sf_sample['alpha'], bins=bins, range=(0, 90))
        g_N_q,  _ = np.histogram(q_sample['alpha'],  bins=bins, range=(0, 90))

        with np.errstate(divide="ignore", invalid="ignore"):
            fq = g_N_q / (g_N_q + g_N_sf)

        fq[np.isnan(fq)] = 0.0
        boot_fq[i] = fq

    return np.nanmean(boot_fq, axis=0), np.nanstd(boot_fq, axis=0)


# %%
def bootstrap_90(data, sf_index, q_index, N, bins=18, seed=None):
    rng = np.random.default_rng(seed)
    boot_fq = np.zeros((N, bins))
    base_idx = data.index
    for i in range(N):
        # inner bootstrap (with replacement)
        sampled_idx = rng.choice(base_idx, size=len(base_idx), replace=True)  # Fixed: removed extra =
        d  = data.loc[sampled_idx]
        
        # Fixed: Need to filter the boolean indices to match sampled data
        # Assuming sf_index and q_index are boolean Series
        sf_mask = sf_index.loc[sampled_idx]
        q_mask = q_index.loc[sampled_idx]
        
        sf_sample = d[sf_mask]
        q_sample  = d[q_mask]
        
        g_N_sf, _ = np.histogram(sf_sample['alpha'], bins=bins, range=(0, 90))
        g_N_q,  _ = np.histogram(q_sample['alpha'],  bins=bins, range=(0, 90))
        
        with np.errstate(divide="ignore", invalid="ignore"):
            fq = g_N_q / (g_N_q + g_N_sf)
        fq[np.isnan(fq)] = 0.0
        boot_fq[i] = fq
    return np.nanmean(boot_fq, axis=0), np.nanstd(boot_fq, axis=0)


# %% jupyter={"outputs_hidden": true}
# set random state and define data 

rng = np.random.default_rng(42)
base_idx = tng100_sfr_info.loc[index].index

# define non-changing MCMC steps

n_walkers_full = 20
n_dim_full = 3 
initial_guess_full = [0.7, 0.025, -3]

for n_sub in [500]:

    for i in range(10):

        # bootstrap n_sub data
        # return mean and std of thr quench fraction every 5 deg.
        
        outer_idx = rng.choice(base_idx,size=n_sub,replace=False)
        
        sub_sfr = tng100_sfr_info.loc[outer_idx]
        sub_sf  = index_sf.loc[outer_idx]
        sub_q   = index_q.loc[outer_idx]
        
        fq_mean_1e6, fq_std_1e6 = bootstrap_90(sub_sfr,sub_sf,sub_q,N=10_000,seed=123)
    
        # MCMC
    
        pos_full = np.array(initial_guess_full) + np.random.randn(n_walkers_full, n_dim_full) * 1e-2
        sampler_full = emcee.EnsembleSampler(n_walkers_full, n_dim_full, calculate_log_probability,args=(angle_array, fq_mean_1e6, fq_std_1e6))
        n_steps = 10_000
        sampler_full.run_mcmc(pos_full, n_steps, progress=True)
        samples_full = sampler_full.get_chain(discard=1000, flat=True)
        log_prob_full = sampler_full.get_log_prob(discard=1000, flat=True)
        mean_params_tng100_all = np.mean(samples_full, axis=0)
        std_params_tng100_all = np.std(samples_full, axis=0)
        a_1e6, b_1e6, f_1e6 = mean_params_tng100_all
        a_1e6_std, b_1e6_std, f_1e6_std = std_params_tng100_all
    
        #
    
        print(np.abs(b_1e6/b_1e6_std))


# %%
def bootstrap_90_optimized(data, sf_index, q_index, N, bins=18, seed=None):
    """
    Optimized bootstrap function - 10-100x faster than original.
    
    Key optimizations:
    1. Vectorized sampling (all N bootstraps at once)
    2. Boolean indexing instead of .loc lookups
    3. Pre-converted indices to numpy arrays
    4. Removed repeated .loc calls in inner loop
    """
    rng = np.random.default_rng(seed)
    boot_fq = np.zeros((N, bins))
    
    # Pre-compute: convert indices to numpy arrays (one-time cost)
    base_idx = data.index.values
    sf_mask = sf_index.reindex(data.index, fill_value=False).values
    q_mask = q_index.reindex(data.index, fill_value=False).values
    alpha_values = data['alpha'].values
    print(q_mask)
    # Vectorized bootstrap sampling
    sampled_indices = rng.choice(len(base_idx), size=(N, len(base_idx)), replace=True)
    
    for i in range(N):
        idx = sampled_indices[i]
        
        # Apply masks to sampled data
        sf_sample_mask = sf_mask[idx]
        q_sample_mask = q_ma
        
    rng = np.random.default_rng(42)
base_idx = tng100_sfr_info.loc[index].index

# define non-changing MCMC steps

n_walkers_full = 20
n_dim_full = 3 
initial_guess_full = [0.7, 0.025, -3]


for n_sub in [1000]:

    for i in range(1):

        # bootstrap n_sub data
        # return mean and std of thr quench fraction every 5 deg.
        
        outer_idx = rng.choice(base_idx,size=n_sub,replace=False)
        
        sub_sfr = tng100_sfr_info.loc[outer_idx]
        sub_sf  = index_sf.loc[outer_idx]
        sub_q   = index_q.loc[outer_idx]
        
        #fq_mean_1e6, fq_std_1e6 = bootstrap_90(sub_sfr,sub_sf,sub_q,N=10_000,seed=123)
        fq_mean_1e6, fq_std_1e6 = bootstrap_90_optimized(sub_sfr, sub_sf, sub_q, N=10_000, seed=123)   
        
        # MCMC
    
        pos_full = np.array(initial_guess_full) + np.random.randn(n_walkers_full, n_dim_full) * 1e-2
        sampler_full = emcee.EnsembleSampler(n_walkers_full, n_dim_full, calculate_log_probability,args=(angle_array, fq_mean_1e6, fq_std_1e6))
        n_steps = 2_000
        sampler_full.run_mcmc(pos_full, n_steps, progress=True)
        samples_full = sampler_full.get_chain(discard=500, flat=True)
        log_prob_full = sampler_full.get_log_prob(discard=500, flat=True)
        mean_params_tng100_all = np.mean(samples_full, axis=0)
        std_params_tng100_all = np.std(samples_full, axis=0)
        a_1e6, b_1e6, f_1e6 = mean_params_tng100_all
        a_1e6_std, b_1e6_std, f_1e6_std = std_params_tng100_all
    
        #
    
        print(np.abs(b_1e6/b_1e6_std))

sk[idx]
        
        # Get alpha values for SF and Q samples
        alpha_sf = alpha_values[idx][sf_sample_mask]
        alpha_q = alpha_values[idx][q_sample_mask]
        
        # Compute histograms
        g_N_sf, _ = np.histogram(alpha_sf, bins=bins, range=(0, 90))
        g_N_q, _ = np.histogram(alpha_q, bins=bins, range=(0, 90))
        
        # Compute fq
        with np.errstate(divide="ignore", invalid="ignore"):
            fq = g_N_q / (g_N_q + g_N_sf)
        fq[np.isnan(fq)] = 0.0
        boot_fq[i] = fq
    
    return np.nanmean(boot_fq, axis=0), np.nanstd(boot_fq, axis=0)

rng = np.random.default_rng(42)
base_idx = tng100_sfr_info.loc[index].index

# define non-changing MCMC steps

n_walkers_full = 20
n_dim_full = 3 
initial_guess_full = [0.7, 0.025, -3]


for n_sub in [1000]:

    for i in range(1):

        # bootstrap n_sub data
        # return mean and std of thr quench fraction every 5 deg.
        
        outer_idx = rng.choice(base_idx,size=n_sub,replace=False)
        
        sub_sfr = tng100_sfr_info.loc[outer_idx]
        sub_sf  = index_sf.loc[outer_idx]
        sub_q   = index_q.loc[outer_idx]
        
        #fq_mean_1e6, fq_std_1e6 = bootstrap_90(sub_sfr,sub_sf,sub_q,N=10_000,seed=123)
        fq_mean_1e6, fq_std_1e6 = bootstrap_90_optimized(sub_sfr, sub_sf, sub_q, N=10_000, seed=123)   
        
        # MCMC
    
        pos_full = np.array(initial_guess_full) + np.random.randn(n_walkers_full, n_dim_full) * 1e-2
        sampler_full = emcee.EnsembleSampler(n_walkers_full, n_dim_full, calculate_log_probability,args=(angle_array, fq_mean_1e6, fq_std_1e6))
        n_steps = 2_000
        sampler_full.run_mcmc(pos_full, n_steps, progress=True)
        samples_full = sampler_full.get_chain(discard=500, flat=True)
        log_prob_full = sampler_full.get_log_prob(discard=500, flat=True)
        mean_params_tng100_all = np.mean(samples_full, axis=0)
        std_params_tng100_all = np.std(samples_full, axis=0)
        a_1e6, b_1e6, f_1e6 = mean_params_tng100_all
        a_1e6_std, b_1e6_std, f_1e6_std = std_params_tng100_all
    
        #
    
        print(np.abs(b_1e6/b_1e6_std))



# %%
def bootstrap_90(data, sf_index, q_index, N, bins=18, seed=None):

    rng = np.random.default_rng(seed)
    boot_fq = np.zeros((N, bins))

    base_idx = data.index

    for i in range(N):

        # inner bootstrap (with replacement)
        sampled_idx = rng.choice(base_idx,=len(base_idx),replace=True)

        d  = data.loc[sampled_idx]
        sf = sf_index.loc[sampled_idx]
        q  = q_index.loc[sampled_idx]

        sf_sample = d.loc[sf]
        q_sample  = d.loc[q]

        g_N_sf, _ = np.histogram(sf_sample['alpha'], bins=bins, range=(0, 90))
        g_N_q,  _ = np.histogram(q_sample['alpha'],  bins=bins, range=(0, 90))

        with np.errstate(divide="ignore", invalid="ignore"):
            fq = g_N_q / (g_N_q + g_N_sf)

        fq[np.isnan(fq)] = 0.0
        boot_fq[i] = fq

    return np.nanmean(boot_fq, axis=0), np.nanstd(boot_fq, axis=0)

# set random state and define data 

rng = np.random.default_rng(42)
base_idx = tng100_sfr_info.loc[index].index

# define non-changing MCMC steps

n_walkers_full = 20
n_dim_full = 3 
initial_guess_full = [0.7, 0.025, -3]

for n_sub in [500]:

    for i in range(1):

        # bootstrap n_sub data
        # return mean and std of thr quench fraction every 5 deg.
        
        outer_idx = rng.choice(base_idx,size=n_sub,replace=False)
        
        sub_sfr = tng100_sfr_info.loc[outer_idx]
        sub_sf  = index_sf.loc[outer_idx]
        sub_q   = index_q.loc[outer_idx]
        
        fq_mean_1e6, fq_std_1e6 = bootstrap_90(sub_sfr,sub_sf,sub_q,N=10_000,seed=123)
    
        # MCMC
    
        pos_full = np.array(initial_guess_full) + np.random.randn(n_walkers_full, n_dim_full) * 1e-2
        sampler_full = emcee.EnsembleSampler(n_walkers_full, n_dim_full, calculate_log_probability,args=(angle_array, fq_mean_1e6, fq_std_1e6))
        n_steps = 10_000
        sampler_full.run_mcmc(pos_full, n_steps, progress=True)
        samples_full = sampler_full.get_chain(discard=1000, flat=True)
        log_prob_full = sampler_full.get_log_prob(discard=1000, flat=True)
        mean_params_tng100_all = np.mean(samples_full, axis=0)
        std_params_tng100_all = np.std(samples_full, axis=0)
        a_1e6, b_1e6, f_1e6 = mean_params_tng100_all
        a_1e6_std, b_1e6_std, f_1e6_std = std_params_tng100_all
    
        #
    
        print(np.abs(b_1e6/b_1e6_std))

# %%
rng = np.random.default_rng(42)
base_idx = tng100_sfr_info.loc[index].index

# define non-changing MCMC steps

n_walkers_full = 20
n_dim_full = 3 
initial_guess_full = [0.7, 0.025, -3]


for n_sub in [1000]:

    for i in range(1):

        # bootstrap n_sub data
        # return mean and std of thr quench fraction every 5 deg.
        
        outer_idx = rng.choice(base_idx,size=n_sub,replace=False)
        
        sub_sfr = tng100_sfr_info.loc[outer_idx]
        sub_sf  = index_sf.loc[outer_idx]
        sub_q   = index_q.loc[outer_idx]
        
        #fq_mean_1e6, fq_std_1e6 = bootstrap_90(sub_sfr,sub_sf,sub_q,N=10_000,seed=123)
        fq_mean_1e6, fq_std_1e6 = bootstrap_90_optimized(sub_sfr, sub_sf, sub_q, N=10_000, seed=123)   
        
        # MCMC
    
        pos_full = np.array(initial_guess_full) + np.random.randn(n_walkers_full, n_dim_full) * 1e-2
        sampler_full = emcee.EnsembleSampler(n_walkers_full, n_dim_full, calculate_log_probability,args=(angle_array, fq_mean_1e6, fq_std_1e6))
        n_steps = 2_000
        sampler_full.run_mcmc(pos_full, n_steps, progress=True)
        samples_full = sampler_full.get_chain(discard=500, flat=True)
        log_prob_full = sampler_full.get_log_prob(discard=500, flat=True)
        mean_params_tng100_all = np.mean(samples_full, axis=0)
        std_params_tng100_all = np.std(samples_full, axis=0)
        a_1e6, b_1e6, f_1e6 = mean_params_tng100_all
        a_1e6_std, b_1e6_std, f_1e6_std = std_params_tng100_all
    
        #
    
        print(np.abs(b_1e6/b_1e6_std))



# %%
from collections import defaultdict
from tqdm import tqdm

def bootstrap_90_optimized(data, sf_index, q_index, N, bins=18, seed=None):
    """
    Optimized bootstrap function - 10-100x faster than original.
    
    Key optimizations:
    1. Vectorized sampling (all N bootstraps at once)
    2. Boolean indexing instead of .loc lookups
    3. Pre-converted indices to numpy arrays
    4. Removed repeated .loc calls in inner loop
    """
    rng = np.random.default_rng(seed)
    boot_fq = np.zeros((N, bins))
    
    # Pre-compute: convert indices to numpy arrays (one-time cost)
    base_idx = data.index.values
    sf_mask = sf_index.reindex(data.index, fill_value=False).values
    q_mask = q_index.reindex(data.index, fill_value=False).values
    alpha_values = data['alpha'].values
    
    # Vectorized bootstrap sampling
    sampled_indices = rng.choice(len(base_idx), size=(N, len(base_idx)), replace=True)
    
    for i in range(N):
        idx = sampled_indices[i]
        
        # Apply masks to sampled data
        sf_sample_mask = sf_mask[idx]
        q_sample_mask = q_mask[idx]
        
        # Get alpha values for SF and Q samples
        alpha_sf = alpha_values[idx][sf_sample_mask]
        alpha_q = alpha_values[idx][q_sample_mask]
        
        # Compute histograms
        g_N_sf, _ = np.histogram(alpha_sf, bins=bins, range=(0, 90))
        g_N_q, _ = np.histogram(alpha_q, bins=bins, range=(0, 90))
        
        # Compute fq
        with np.errstate(divide="ignore", invalid="ignore"):
            fq = g_N_q / (g_N_q + g_N_sf)
        fq[np.isnan(fq)] = 0.0
        boot_fq[i] = fq
    
    return np.nanmean(boot_fq, axis=0), np.nanstd(boot_fq, axis=0)


rng = np.random.default_rng(42)
base_idx = tng100_sfr_info.loc[index].index

# Define non-changing MCMC steps
n_walkers_full = 20
n_dim_full = 3 
initial_guess_full = [0.7, 0.025, -3]
n_steps = 2_000

# Store all results
all_results = defaultdict(list)

# Loop over different sample sizes
for n_sub in [100, 1000, 10000]:
    print(f"\n{'='*60}")
    print(f"Running n_sub = {n_sub}")
    print(f"{'='*60}")
    
    # Multiple iterations for this n_sub
    n_iterations = 1000  # Change this to however many you want
    
    for iteration in tqdm(range(n_iterations)):
        #print(f"\n  Iteration {iteration+1}/{n_iterations} (n_sub={n_sub})")
        
        # Outer bootstrap: sample n_sub data points
        outer_idx = rng.choice(base_idx, size=n_sub, replace=False)
        
        sub_sfr = tng100_sfr_info.loc[outer_idx]
        sub_sf  = index_sf.loc[outer_idx]
        sub_q   = index_q.loc[outer_idx]
        
        # Inner bootstrap with unique seed per iteration
        seed = 123 + n_sub + iteration  # Unique seed for each combination
        fq_mean, fq_std = bootstrap_90_optimized(sub_sfr, sub_sf, sub_q, N=10_000, seed=seed)
        
        # MCMC
        pos_full = np.array(initial_guess_full) + np.random.randn(n_walkers_full, n_dim_full) * 1e-2
        sampler_full = emcee.EnsembleSampler(n_walkers_full, n_dim_full, calculate_log_probability,args=(angle_array, fq_mean, fq_std))
        sampler_full.run_mcmc(pos_full, n_steps, progress=False)  # Set to True if you want progress bars
        
        samples_full = sampler_full.get_chain(discard=500, flat=True)
        
        mean_params = np.mean(samples_full, axis=0)
        std_params = np.std(samples_full, axis=0)
        
        a, b, f = mean_params
        a_std, b_std, f_std = std_params
        
        b_ratio = np.abs(b / b_std)
        
        # Store results
        all_results[n_sub].append({'iteration': iteration,'n_sub': n_sub,'b_ratio': b_ratio,'params': mean_params,
            'std_params': std_params,'a': a, 'b': b, 'f': f,'a_std': a_std, 'b_std': b_std, 'f_std': f_std,'fq_mean': fq_mean,
            'fq_std': fq_std})
        
        #print(f"    b_ratio: {b_ratio:.4f}, b: {b:.6f} ± {b_std:.6f}")

# %%
print(f"\n{'='*60}")
print("SUMMARY OF RESULTS")
print(f"{'='*60}\n")

for n_sub in [100, 1000, 10000]:
    results = all_results[n_sub]
    
    b_ratios = [r['b_ratio'] for r in results]
    b_values = [r['b'] for r in results]
    b_stds = [r['b_std'] for r in results]
    
    print(f"n_sub = {n_sub}:")
    print(f"  b_ratio:  {np.mean(b_ratios):.4f} ± {np.std(b_ratios):.4f}")
    print(f"  b:        {np.mean(b_values):.6f} ± {np.std(b_values):.6f}")
    print(f"  b_std:    {np.mean(b_stds):.6f} ± {np.std(b_stds):.6f}")
    print()


# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, n_sub in enumerate([100, 1000, 10000]):
    results = all_results[n_sub]
    b_ratios = [r['b_ratio'] for r in results]
    
    axes[idx].hist(b_ratios, bins=10, edgecolor='black', alpha=0.7)
    axes[idx].axvline(np.mean(b_ratios), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(b_ratios):.2f}')
    axes[idx].set_xlabel('b_ratio')
    axes[idx].set_ylabel('Frequency')
    axes[idx].set_title(f'n_sub = {n_sub}')
    axes[idx].legend()

# %%
base_idx

sampled_idx = rng.choice(base_idx, size=len(base_idx), replace=True)
d = base_idx.loc[sampled_idx]

print("Number of rows in d:", len(d))
print("Number of unique indices:", d.index.nunique())


# %%
def bootstrap_90_exact(alpha, sf_mask, q_mask, bins=18, N=10_000, seed=None):

    rng = np.random.default_rng(seed)

    bin_edges = np.linspace(0, 90, bins + 1)
    bin_idx = np.digitize(alpha, bin_edges) - 1

    valid = (bin_idx >= 0) & (bin_idx < bins)
    bin_idx = bin_idx[valid]
    sf_mask = sf_mask[valid]
    q_mask  = q_mask[valid]

    n = len(bin_idx)
    boot_fq = np.zeros((N, bins))

    for i in range(N):
        idx = rng.integers(0, n, size=n)

        b = bin_idx[idx]
        sf = sf_mask[idx]
        q  = q_mask[idx]

        sf_counts = np.bincount(b[sf], minlength=bins)
        q_counts  = np.bincount(b[q],  minlength=bins)

        with np.errstate(divide="ignore", invalid="ignore"):
            fq = q_counts / (q_counts + sf_counts)

        fq[np.isnan(fq)] = 0.0
        boot_fq[i] = fq

    return boot_fq.mean(axis=0), boot_fq.std(axis=0)


# %%
def bootstrap_90_exact(alpha, sf_mask, q_mask, bins=18, N=10_000, seed=None):
    rng = np.random.default_rng(seed)

    bin_edges = np.linspace(0, 90, bins + 1)
    n = len(alpha)
    boot_fq = np.zeros((N, bins))

    for i in range(N):
        # resample row indices with replacement
        sampled_idx = rng.integers(0, n, size=n)

        alpha_samp = alpha[sampled_idx]
        sf_samp    = sf_mask[sampled_idx]
        q_samp     = q_mask[sampled_idx]

        # use np.histogram (identical to original)
        g_N_sf, _ = np.histogram(alpha_samp[sf_samp], bins=bins, range=(0,90))
        g_N_q,  _ = np.histogram(alpha_samp[q_samp],  bins=bins, range=(0,90))

        with np.errstate(divide='ignore', invalid='ignore'):
            fq = g_N_q / (g_N_q + g_N_sf)
        fq[np.isnan(fq)] = 0.0
        boot_fq[i] = fq

    return boot_fq.mean(axis=0), boot_fq.std(axis=0)


# %%
fq_slow_mean, fq_slow_std = bootstrap_90(
    sub_sfr, sub_sf, sub_q, N=2000
)

fq_fast_mean, fq_fast_std = bootstrap_90_fast(
    alpha, sf, q, N=2000
)

fq_exact_mean, fq_exact_std = bootstrap_90_exact(
    alpha, sf, q, N=2000
)


plt.plot(fq_slow_mean, label="slow")
plt.plot(fq_fast_mean, '--', label="fast")
plt.plot(fq_exact_mean, '--', label="exact")

plt.legend()


# %%
sampled_idx = rng.choice(base_idx, size=len(base_idx), replace=True)
d = data.loc[sampled_idx]

print("Number of rows in d:", len(d))
print("Number of unique indices:", d.index.nunique())

# %%
# boostrap TNG100 data 

#fq_mean_1e8, fq_std_1e8 = boostrap_90(tng100_sfr_info[index],index_sf,index_q,10000)
#fq_mean_1e7, fq_std_1e7 = boostrap_90(tng100_sfr_info[index],index_sf,index_q,10000)

rng = np.random.default_rng(42) # set random seed

subsample = rng.choice(tng100_sfr_info[index], size=2, replace=True)

#fq_mean_1e6, fq_std_1e6 = boostrap_90(tng100_sfr_info[index],index_sf,index_q,10000)

# %%
def calculate_log_likelihood(theta, bin_centers, f_q, sigma_i):

    a, b, f = theta

    # Compute total variance
    s_i = sigma_i**2 + (np.exp(f))**2

    # Model prediction
    f_model = a + b * np.cos(2 * np.radians(bin_centers))

    # Residuals and log-likelihood
    residuals = (f_q - f_model)**2 / s_i
    log_likelihood = -0.5 * np.sum(residuals + np.log(2 * np.pi * s_i))
    
    return log_likelihood

def log_prior(theta):
    a, b, f = theta
    if 0 < a < 1 and -1 < b < 1 and -10 < f < 2:
        return 0.0
    return -np.inf

def calculate_log_probability(theta, bin_centers, f_q, sigma_i):
    
    log_prior_val = log_prior(theta)
    if not np.isfinite(log_prior_val):
        return -np.inf
    log_likelihood = calculate_log_likelihood(theta, bin_centers, f_q, sigma_i)
    return log_prior_val + log_likelihood


# %%
# 1e6

# MCMC Configuration

n_walkers_full = 20
n_dim_full = 3 
initial_guess_full = [0.7, 0.025, -3]
pos_full = np.array(initial_guess_full) + np.random.randn(n_walkers_full, n_dim_full) * 1e-2

# Setting up the sampler
sampler_full = emcee.EnsembleSampler(n_walkers_full, n_dim_full, calculate_log_probability,args=(angle_array, fq_mean_1e6, fq_std_1e6))

# Run MCMC
n_steps = 10_000
sampler_full.run_mcmc(pos_full, n_steps, progress=True)

# Post-Processing
samples_full = sampler_full.get_chain(discard=1000, flat=True)
log_prob_full = sampler_full.get_log_prob(discard=1000, flat=True)

mean_params_tng100_all = np.mean(samples_full, axis=0)
std_params_tng100_all = np.std(samples_full, axis=0)
a_1e6, b_1e6, f_1e6 = mean_params_tng100_all
a_1e6_std, b_1e6_std, f_1e6_std = std_params_tng100_all

# 1e7

# MCMC Configuration

n_walkers_full = 20
n_dim_full = 3 
initial_guess_full = [0.7, 0.025, -3]
pos_full = np.array(initial_guess_full) + np.random.randn(n_walkers_full, n_dim_full) * 1e-2

# Setting up the sampler
sampler_full = emcee.EnsembleSampler(n_walkers_full, n_dim_full, calculate_log_probability,args=(angle_array, fq_mean_1e7, fq_std_1e7))

# Run MCMC
n_steps = 10_000
sampler_full.run_mcmc(pos_full, n_steps, progress=True)

# Post-Processing
samples_full = sampler_full.get_chain(discard=1000, flat=True)
log_prob_full = sampler_full.get_log_prob(discard=1000, flat=True)

mean_params_tng100_all = np.mean(samples_full, axis=0)
std_params_tng100_all = np.std(samples_full, axis=0)
a_1e7, b_1e7, f_1e7 = mean_params_tng100_all
a_1e7_std, b_1e7_std, f_1e7_std = std_params_tng100_all

# 1e8

# MCMC Configuration

n_walkers_full = 20
n_dim_full = 3 
initial_guess_full = [0.7, 0.025, -3]
pos_full = np.array(initial_guess_full) + np.random.randn(n_walkers_full, n_dim_full) * 1e-2

# Setting up the sampler
sampler_full = emcee.EnsembleSampler(n_walkers_full, n_dim_full, calculate_log_probability,args=(angle_array, fq_mean_1e8, fq_std_1e8))

# Run MCMC
n_steps = 10_000
sampler_full.run_mcmc(pos_full, n_steps, progress=True)

# Post-Processing
samples_full = sampler_full.get_chain(discard=1000, flat=True)
log_prob_full = sampler_full.get_log_prob(discard=1000, flat=True)

mean_params_tng100_all = np.mean(samples_full, axis=0)
std_params_tng100_all = np.std(samples_full, axis=0)
a_1e8, b_1e8, f_1e8 = mean_params_tng100_all
a_1e8_std, b_1e8_std, f_1e8_std = std_params_tng100_all

# %%
# re-fit SAGA and ELVES data with updated boostrapping 

# SAGA

# MCMC Configuration

n_walkers_full = 20
n_dim_full = 3 
initial_guess_full = [0.7, 0.025, -3]
pos_full = np.array(initial_guess_full) + np.random.randn(n_walkers_full, n_dim_full) * 1e-2

# Setting up the sampler
sampler_full = emcee.EnsembleSampler(n_walkers_full, n_dim_full, calculate_log_probability,args=(angle_array, fq_mean_saga, fq_std_saga))

# Run MCMC
n_steps = 10_000
sampler_full.run_mcmc(pos_full, n_steps, progress=True)

# Post-Processing
samples_full = sampler_full.get_chain(discard=1000, flat=True)
log_prob_full = sampler_full.get_log_prob(discard=1000, flat=True)

mean_params_tng100_all = np.mean(samples_full, axis=0)
std_params_tng100_all = np.std(samples_full, axis=0)
a_saga, b_saga, f_saga = mean_params_tng100_all
a_saga_std, b_saga_std, f_saga_std = std_params_tng100_all

# ELVES 

# MCMC Configuration

n_walkers_full = 20
n_dim_full = 3 
initial_guess_full = [0.7, 0.025, -3]
pos_full = np.array(initial_guess_full) + np.random.randn(n_walkers_full, n_dim_full) * 1e-2

# Setting up the sampler
sampler_full = emcee.EnsembleSampler(n_walkers_full, n_dim_full, calculate_log_probability,args=(angle_array, fq_mean_elves, fq_std_elves))

# Run MCMC
n_steps = 10_000
sampler_full.run_mcmc(pos_full, n_steps, progress=True)

# Post-Processing
samples_full = sampler_full.get_chain(discard=1000, flat=True)
log_prob_full = sampler_full.get_log_prob(discard=1000, flat=True)

mean_params_tng100_all = np.mean(samples_full, axis=0)
std_params_tng100_all = np.std(samples_full, axis=0)
a_elves, b_elves, f_elves = mean_params_tng100_all
a_elves_std, b_elves_std, f_elves_std = std_params_tng100_all

# %%
# New high-contrast colors
saga_line = '#1f77b4'
saga_fill = '#aec7e8'
elves_line = '#d62728'
elves_fill = '#ff9896'

fig, (ax_hist,ax_hist2) = plt.subplots(1, 2, figsize=(13, 5))

# SDSS, TNG, SMDPL (from Michael)

#sdss_fq = np.loadtxt('../../sdss/figure1/sdss_fq.txt')
#tng100_fq = np.loadtxt('../../sdss/figure1/tng100_fq.txt')
#smdpl_fq = np.loadtxt('../../sdss/figure1/smdpl_fq.txt')

ax_hist.errorbar(angle_array, fq_mean_1e8, yerr=fq_std_1e8,fmt='o', color='black', mfc='black', mec='black', mew=1,capsize=3)
ax_hist.errorbar(angle_array, fq_mean_1e7, yerr=fq_std_1e7,fmt='o', color='black', mfc='black', mec='black', mew=1,capsize=3)
ax_hist.errorbar(angle_array, fq_mean_1e6, yerr=fq_std_1e6,fmt='o', color='black', mfc='black', mec='black', mew=1,capsize=3)

x = np.linspace(0,np.pi/2,1000)
ax_hist.plot((x*u.rad).to('degree'),a_1e6 + b_1e6 * np.cos(2 * x),color='black', lw=2,ls=':')
ax_hist.plot((x*u.rad).to('degree'),a_1e7 + b_1e7 * np.cos(2 * x),color='black', lw=2,ls='--')
ax_hist.plot((x*u.rad).to('degree'),a_1e8 + b_1e8 * np.cos(2 * x),color='black', lw=2,ls='-')

n_mc = 10000
a_samp = np.random.normal(a_1e6, a_1e6_std, n_mc)
b_samp = np.random.normal(b_1e6, b_1e6_std, n_mc)
cos2x = np.cos(2 * x)
y_mc = a_samp[:, None] + b_samp[:, None] * cos2x[None, :]
y_med   = np.percentile(y_mc, 50, axis=0)
y_low   = np.percentile(y_mc, 16, axis=0)
y_high  = np.percentile(y_mc, 84, axis=0)
theta_deg = (x * u.rad).to('degree').value
ax_hist.fill_between(theta_deg,y_low,y_high,color='k',alpha=0.1,edgecolor=None)

n_mc = 10000
a_samp = np.random.normal(a_1e7, a_1e7_std, n_mc)
b_samp = np.random.normal(b_1e7, b_1e7_std, n_mc)
cos2x = np.cos(2 * x)
y_mc = a_samp[:, None] + b_samp[:, None] * cos2x[None, :]
y_med   = np.percentile(y_mc, 50, axis=0)
y_low   = np.percentile(y_mc, 16, axis=0)
y_high  = np.percentile(y_mc, 84, axis=0)
theta_deg = (x * u.rad).to('degree').value
ax_hist.fill_between(theta_deg,y_low,y_high,color='k',alpha=0.1,edgecolor=None)

n_mc = 10000
a_samp = np.random.normal(a_1e8, a_1e8_std, n_mc)
b_samp = np.random.normal(b_1e8, b_1e8_std, n_mc)
cos2x = np.cos(2 * x)
y_mc = a_samp[:, None] + b_samp[:, None] * cos2x[None, :]
y_med   = np.percentile(y_mc, 50, axis=0)
y_low   = np.percentile(y_mc, 16, axis=0)
y_high  = np.percentile(y_mc, 84, axis=0)
theta_deg = (x * u.rad).to('degree').value
ax_hist.fill_between(theta_deg,y_low,y_high,color='k',alpha=0.1,edgecolor=None)



ax_hist.set_ylabel(r'$\mathrm{f_q}$', fontsize=14, labelpad=20)
ax_hist.set_xlabel(r'$\theta$ [deg.]', fontsize=14, labelpad=20)
ax_hist2.set_xlabel(r'$\theta$ [deg.]', fontsize=14, labelpad=20)

ax_hist.set_xlim(0, 90)
ax_hist2.set_xlim(0, 90)

ax_hist.set_ylim(0, 1)
ax_hist2.set_ylim(0, 1)

ax_hist.tick_params(axis="both", which="major", direction="in", labelsize=12, length=7, width=1)
ax_hist.tick_params(axis="both", which="minor", direction="in", labelsize=12, length=2, width=1)

ax_hist2.tick_params(axis="both", which="major", direction="in", labelsize=12, length=7, width=1)
ax_hist2.tick_params(axis="both", which="minor", direction="in", labelsize=12, length=2, width=1)

ax_hist2.set_yticks([])

plt.subplots_adjust(wspace=0.05)
plt.show()

# %% [markdown]
# ## BIC/AIC Analysis:

# %%

# %% [markdown]
# ### Interpretation: If the amplitude b = 0, then the sinsoidal function is merely a constant, or the quench fraction. We can therefore ask, how many std away from 0 is the parameter b mean? We find that there is significant overlap in the SAGA and ELVES b parameter. However, there is less overlap in the TNG100-1 subsamples.

# %%
# calulate how many std the mean is from 0 (for the parameter b, such that b = 0 is consistent with a st)

n_mc = 100000
b_samp = np.random.normal(b_saga, b_saga_std, n_mc)

#plt.hist(b_samp,bins=100,histtype='step',color='k');

b_mean = np.mean(b_samp)
b_std  = np.std(b_samp)
sigma_significance = np.abs(b_mean / b_std)
print(sigma_significance)

#######

b_samp = np.random.normal(b_elves, b_elves_std, n_mc)

#plt.hist(b_samp,bins=100,histtype='step',color='k');

b_mean = np.mean(b_samp)
b_std  = np.std(b_samp)
sigma_significance = np.abs(b_mean / b_std)
print(sigma_significance)

#######

b_samp = np.random.normal(b_1e8, b_1e8_std, n_mc)

#plt.hist(b_samp,bins=100,histtype='step',color='k');

b_mean = np.mean(b_samp)
b_std  = np.std(b_samp)
sigma_significance = np.abs(b_mean / b_std)
print(sigma_significance)

#######

b_samp = np.random.normal(b_1e7, b_1e7_std, n_mc)

#plt.hist(b_samp,bins=100,histtype='step',color='k');

b_mean = np.mean(b_samp)
b_std  = np.std(b_samp)
sigma_significance = np.abs(b_mean / b_std)
print(sigma_significance)

#######

b_samp = np.random.normal(b_1e6, b_1e6_std, n_mc)

#plt.hist(b_samp,bins=100,histtype='step',color='k');

b_mean = np.mean(b_samp)
b_std  = np.std(b_samp)
sigma_significance = np.abs(b_mean / b_std)
print(sigma_significance)


# %%
# calculate the BIC (and AIC) for the sinusoidal fit versus constant
# be careful of f value -- is this an extra parmameter?

def sinusoid(x, a, b, f):
    
    return a + b * np.cos(2 * x)

def BIC(x_data, y_data, y_err, a_fit, b_fit, f_fit):

    resid = y_data - sinusoid(x_data, a_fit, b_fit, f_fit)
    print(resid)                       
    n = len(y_data)
    k = 2
    logL = -0.5 * np.sum((resid / y_err)**2 + np.log(2*np.pi*y_err**2))
    bic_sin = k * np.log(n) - 2 * logL
    print("BIC Sinudoid Fit =", bic_sin) 
    aic_sin = 2*k - 2*logL
    print("AIC Sinudoid Fit =", aic_sin) 

    resid_const = y_data - np.mean(y_data)
    k_const = 1
    logL_const = -0.5 * np.sum((resid_const / y_err)**2 + np.log(2*np.pi*y_err**2))
    bic_const = k_const * np.log(n) - 2 * logL_const
    print("BIC Constant Fit =", bic_const)
    aic_const = 2*k - 2*logL_const
    print("AIC Constant Fit =", aic_const) 

    print('Delta BIC =',np.abs(bic_const-bic_sin))
    print('Delta AIC =',np.abs(aic_const-aic_sin))
    if bic_const < bic_sin:
        print("From BIC, Constant model preferred")
    else:
        print("From BIC, Sinusoid model preferred")

    if aic_const < aic_sin:
        print("From AIC, Constant model preferred")
    else:
        print("From AIC, Sinusoid model preferred")

print('ELVES:')
BIC(angle_array, fq_mean_elves, fq_std_elves, a_elves, b_elves, f_elves)
print(' ')
print('SAGA:')
BIC(angle_array, fq_mean_saga, fq_std_saga, a_saga, b_saga, f_saga)
print(' ')
print('TNG100-1, > 1e8:')
BIC(angle_array, fq_mean_1e8, fq_std_1e8, a_1e8, b_1e8, f_1e8)
print(' ')
print('TNG100-1, > 1e7:')
BIC(angle_array, fq_mean_1e7, fq_std_1e7, a_1e7, b_1e7, f_1e7)
print(' ')
print('TNG100-1, > 1e6:')
BIC(angle_array, fq_mean_1e6, fq_std_1e6, a_1e6, b_1e6, f_1e6)

# %% [markdown]
# ### Interpretation: Instead of looking at the amplitude b, we can instead ask: How much better of a fit is the sinusoidal function compared to a constant (the mean quench fraction across all angle bins). This can be calculated using both the BIC (penalizes number of paramaeters more) and AIC. We find that all data are better fit with a constant. Supposedly the AIC for the SAGA data is slightly lower for the sinusoid compared to the constant model, but the $\Delta$BIC is 1.15. As a good rule of thumb, > 10 suggests that there is strong evidence.

# %% [markdown]
# ## Old Code from Nick:

# %% [markdown]
# ## Alignment Test

# %%

#SAGA
theta_deg = np.array(saga_90)  # your folded angles
theta_rad = np.deg2rad(theta_deg)

# Effect size
A_obs = np.mean(np.cos(2 * theta_rad))

# Bootstrap CI
B = 5000
rng = np.random.default_rng(0)
A_boot = []
n = len(theta_rad)

for _ in range(B):
    sample = rng.choice(theta_rad, size=n, replace=True)
    A_boot.append(np.mean(np.cos(2 * sample)))

lo, hi = np.percentile(A_boot, [2.5, 97.5])
print("SAGA")
print(f"<cos(2θ)> = {A_obs:.4f}")
print(f"95% CI: [{lo:.4f}, {hi:.4f}]")


# %%

#ELVES
theta_deg = np.array(elves_90)  # your folded angles
theta_rad = np.deg2rad(theta_deg)

# Effect size
A_obs = np.mean(np.cos(2 * theta_rad))

# Bootstrap CI
B = 5000
rng = np.random.default_rng(0)
A_boot = []
n = len(theta_rad)

for _ in range(B):
    sample = rng.choice(theta_rad, size=n, replace=True)
    A_boot.append(np.mean(np.cos(2 * sample)))

lo, hi = np.percentile(A_boot, [2.5, 97.5])
print("ELVES")
print(f"<cos(2θ)> = {A_obs:.4f}")
print(f"95% CI: [{lo:.4f}, {hi:.4f}]")

# %%
import numpy as np
from scipy.stats import ks_2samp
from scipy.stats import kstest

print("SAGA: Testing Whether Quench and Star forming follow different distribution")
# Inputs:
# theta_deg: folded angles in [0, 90], shape (N,)
# quenched:  1 for quenched, 0 for star-forming, shape (N,)
# host_id:   host identifier per satellite, shape (N,)
theta_deg = np.asarray(saga_90)
quenched  = np.asarray(saga_quenched).astype(int)
host_id   = np.asarray(saga_sats['HOSTID'][radius_limit])


# Split groups
th_q  = theta_deg[quenched == 1]
th_sf = theta_deg[quenched == 0]


print("saga:", kstest(theta_deg, 'uniform' , args=(0, 90)))


# --- (1) Two-sample KS ---
D, p_ks = ks_2samp(th_q, th_sf, alternative='two-sided')
# Extract scalar values to avoid formatting errors
D = D.item() if hasattr(D, 'item') else (float(D[0]) if hasattr(D, '__len__') else D)
p_ks = p_ks.item() if hasattr(p_ks, 'item') else (float(p_ks[0]) if hasattr(p_ks, '__len__') else p_ks)



# --- (2) ΔA effect size ---
c2 = np.cos(2 * np.deg2rad(theta_deg))
A_q  = c2[quenched == 1].mean()
A_sf = c2[quenched == 0].mean()
dA   = A_q - A_sf

print(f"KS: D={D:.4f}, p={p_ks:.3g}")
print(f"A_q={A_q:.4f}, A_sf={A_sf:.4f}, ΔA={dA:.4f}")

# --- (3) 95% CI for ΔA via cluster (host) bootstrap ---
rng = np.random.default_rng(0)
hosts = np.unique(host_id)
B = 5000
dA_boot = []

for _ in range(B):
    boot_hosts = rng.choice(hosts, size=len(hosts), replace=True)
    # keep all satellites belonging to sampled hosts
    mask = np.isin(host_id, boot_hosts)
    c2_b = c2[mask]
    q_b  = quenched[mask]
    if (q_b == 1).any() and (q_b == 0).any():
        dA_boot.append(c2_b[q_b==1].mean() - c2_b[q_b==0].mean())

lo, hi = np.percentile(dA_boot, [2.5, 97.5])
print(f"ΔA 95% CI: [{lo:.4f}, {hi:.4f}]")

# --- (4) Cluster-aware permutation p-values ---
# Permute labels WITHIN each host; preserves per-host clustering and class sizes.
def permute_labels_within_host(q, h, rng):
    q_perm = q.copy()
    for hh in np.unique(h):
        idx = np.where(h == hh)[0]
        q_perm[idx] = rng.permutation(q_perm[idx])
    return q_perm

Bperm = 5000
dA_perm = []
D_perm  = []

for _ in range(Bperm):
    q_p = permute_labels_within_host(quenched, host_id, rng)
    # ΔA under permuted labels
    A_q_p  = c2[q_p==1].mean() if (q_p==1).any() else np.nan
    A_sf_p = c2[q_p==0].mean() if (q_p==0).any() else np.nan
    dA_perm.append(A_q_p - A_sf_p)

    # KS under permuted labels
    th_q_p  = theta_deg[q_p == 1]
    th_sf_p = theta_deg[q_p == 0]
    if len(th_q_p) > 0 and len(th_sf_p) > 0:
        Dp, _ = ks_2samp(th_q_p, th_sf_p, alternative='two-sided')
        D_perm.append(Dp.item() if hasattr(Dp, 'item') else (float(Dp[0]) if hasattr(Dp, '__len__') else Dp))

dA_perm = np.array(dA_perm, dtype=float)
D_perm  = np.array(D_perm, dtype=float)

p_perm_dA = (np.sum(np.abs(dA_perm) >= abs(dA)) + 1) / (np.sum(~np.isnan(dA_perm)) + 1)
p_perm_KS = (np.sum(D_perm >= D) + 1) / (len(D_perm) + 1)

print(f"Permutation p (ΔA): {p_perm_dA:.4g}")
print(f"Permutation p (KS): {p_perm_KS:.4g}")


# %%
import numpy as np
from scipy.stats import ks_2samp
from scipy.stats import kstest

print("ELVES: Testing Whether Quench and Star forming follow different distribution")
# Inputs:
# theta_deg: folded angles in [0, 90], shape (N,)
# quenched:  1 for quenched, 0 for star-forming, shape (N,)
# host_id:   host identifier per satellite, shape (N,)
theta_deg = np.asarray(elves_90)
quenched  = np.asarray(elves_quenched).astype(int)
# Use the kept_indices from the ELVES processing to get the correct host IDs
host_id   = np.asarray(sats['Host'].iloc[kept_indices])


# Split groups
th_q  = theta_deg[quenched == 1]
th_sf = theta_deg[quenched == 0]


print("ELVES:", kstest(theta_deg, 'uniform' , args=(0, 90)))


# --- (1) Two-sample KS ---
D, p_ks = ks_2samp(th_q, th_sf, alternative='two-sided')
# Extract scalar values to avoid formatting errors
D = D.item() if hasattr(D, 'item') else (float(D[0]) if hasattr(D, '__len__') else D)
p_ks = p_ks.item() if hasattr(p_ks, 'item') else (float(p_ks[0]) if hasattr(p_ks, '__len__') else p_ks)



# --- (2) ΔA effect size ---
c2 = np.cos(2 * np.deg2rad(theta_deg))
A_q  = c2[quenched == 1].mean()
A_sf = c2[quenched == 0].mean()
dA   = A_q - A_sf

print(f"KS: D={D:.4f}, p={p_ks:.3g}")
print(f"A_q={A_q:.4f}, A_sf={A_sf:.4f}, ΔA={dA:.4f}")

# --- (3) 95% CI for ΔA via cluster (host) bootstrap ---
rng = np.random.default_rng(0)
hosts = np.unique(host_id)
B = 5000
dA_boot = []

for _ in range(B):
    boot_hosts = rng.choice(hosts, size=len(hosts), replace=True)
    # keep all satellites belonging to sampled hosts
    mask = np.isin(host_id, boot_hosts)
    c2_b = c2[mask]
    q_b  = quenched[mask]
    if (q_b == 1).any() and (q_b == 0).any():
        dA_boot.append(c2_b[q_b==1].mean() - c2_b[q_b==0].mean())

lo, hi = np.percentile(dA_boot, [2.5, 97.5])
print(f"ΔA 95% CI: [{lo:.4f}, {hi:.4f}]")

# --- (4) Cluster-aware permutation p-values ---
# Permute labels WITHIN each host; preserves per-host clustering and class sizes.
def permute_labels_within_host(q, h, rng):
    q_perm = q.copy()
    for hh in np.unique(h):
        idx = np.where(h == hh)[0]
        q_perm[idx] = rng.permutation(q_perm[idx])
    return q_perm

Bperm = 5000
dA_perm = []
D_perm  = []

for _ in range(Bperm):
    q_p = permute_labels_within_host(quenched, host_id, rng)
    # ΔA under permuted labels
    A_q_p  = c2[q_p==1].mean() if (q_p==1).any() else np.nan
    A_sf_p = c2[q_p==0].mean() if (q_p==0).any() else np.nan
    dA_perm.append(A_q_p - A_sf_p)

    # KS under permuted labels
    th_q_p  = theta_deg[q_p == 1]
    th_sf_p = theta_deg[q_p == 0]
    if len(th_q_p) > 0 and len(th_sf_p) > 0:
        Dp, _ = ks_2samp(th_q_p, th_sf_p, alternative='two-sided')
        D_perm.append(Dp.item() if hasattr(Dp, 'item') else (float(Dp[0]) if hasattr(Dp, '__len__') else Dp))

dA_perm = np.array(dA_perm, dtype=float)
D_perm  = np.array(D_perm, dtype=float)

p_perm_dA = (np.sum(np.abs(dA_perm) >= abs(dA)) + 1) / (np.sum(~np.isnan(dA_perm)) + 1)
p_perm_KS = (np.sum(D_perm >= D) + 1) / (len(D_perm) + 1)

print(f"Permutation p (ΔA): {p_perm_dA:.4g}")
print(f"Permutation p (KS): {p_perm_KS:.4g}")


# %%

# %%

# %%
