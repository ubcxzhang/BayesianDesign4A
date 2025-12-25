import cupy as cp
import numpy as np
import argparse
import ast
import re
import os

from scipy.stats import norm
from gbayesdesign.BayesSampler import BayesSampler
from gbayesdesign.mvn import van_der_corput
from gbayesdesign.powerZ import power, constraint, power_1d, constraint_1d
from gbayesdesign.Optimizer import ZSQP, ZSQP_1d

def pyramid(n):
    return int((n**2+n)/2)

def xtPermutator(values):
    """
    Permutate Xt array into pairs where i <= j
    """
    # Handles single-value input file
    if values.shape == ():
        return np.array([[values,values]])

    VLENGTH = values.shape[0]
    # Set pyramid
    xtSet_array = np.zeros(shape=(pyramid(VLENGTH),2))
    counter = 0
    for i in range(VLENGTH):
        for j in range(VLENGTH):
            # print(values[i],values[j],values[i]<=values[j])
            if (values[i] <= values[j]):
                xtSet_array[counter] = np.array([values[i],values[j]])
                counter += 1
    return np.array(xtSet_array)

def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument('-i', '--inputPath', 
        type=str, default='input/default_table_none_true.csv')
    parser.add_argument('-o', '--outputPath', 
        type=str, default='results/sub')
    parser.add_argument('-v', '--verbose', action='store_true')

    # test input selections
    parser.add_argument('-T', '--test_iteration', type=int, default=0)
    parser.add_argument('-es','--effect_setting',
                        type=int, default=2, choices=range(0, 3)) # 0: none, 1: week, 2: strong
    parser.add_argument('-tt','--test_type',
                        type=str, default='generic')
    parser.add_argument('-nb','--num_batches',
                        type=int, default=0)

    args = parser.parse_args(arguments)
    return args

if '__main__':
    import pandas as pd 
    import time

    NMAX = 6000;
    SLICE_SIZE = 20000;
    N_DELTAXT = 80000;
    ALPHA = 0.025
    SIGMA1_COEFF = 0.01
    SEED = 101


    args = parse_arguments()
    TEST_TYPE = args.test_type
    TEST_ITERATION = args.test_iteration
    # EFFECT = 'strong'
    # Default effect setting is 'strong'
    if args.effect_setting == 0:
        EFFECT  = 'none'
    elif args.effect_setting == 1:
        EFFECT  = 'weak'
    elif args.effect_setting == 2:
        EFFECT  = 'strong'
    # elif args.effect_setting == 3:
    #     EFFECT  = ['none', 'weak', 'strong']
    else:
        EFFECT  = 'strong'

    SORT_COLUMN = 'r'

    args = parse_arguments()
    INPUT_PATH = args.inputPath
    OUTPUT_PATH = args.outputPath
    TEST_ITERATION = args.test_iteration
    IS_VERBOSE = args.verbose
    NUM_BATCHES = args.num_batches

    inputTable = pd.read_csv(INPUT_PATH, index_col=0)

    lattice_points = van_der_corput(NMAX, scramble=True, seed=SEED)
    lattice_points_col=lattice_points.reshape(-1,1)

    Z = cp.array([1.,1.5])
    supplement_list = []
    all_results_data = []

    # Save combined results DataFrame to a single CSV
    results_csv_path = os.path.join(OUTPUT_PATH, f'{TEST_TYPE}Test_i{TEST_ITERATION}_{EFFECT}_combined_results_01.csv')
    # Start the loop to process rows and save the DataFrame every 2 hours
    total_start_time = time.time()
    save_interval = 2 * 60 * 60 - 5 * 60# 2 hours in seconds
    
    num_subgroups = NUM_BATCHES
    if num_subgroups == 0:
        selected_subgroup = range(inputTable.shape[0])
    else:
        indices = np.arange(inputTable.shape[0])
        subgroups = np.array_split(indices, num_subgroups)
        if 0 <= TEST_ITERATION < num_subgroups:
            selected_subgroup = subgroups[TEST_ITERATION]
    for index in selected_subgroup:
        print("DataFrame Index:", index)
        t = inputTable.at[index, 't']
        r = inputTable.at[index, 'r']
        Is = inputTable.at[index, 'Is']
        p_1 = inputTable.at[index, 'p_1']
        # diff = inputTable.at[index, 'diff']
        x1 = inputTable.at[index, 'X1_t']
        x1_orig = inputTable.at[index, 'X1_t_orig'] if 'X1_t_orig' in inputTable.columns else np.nan
        x2 = inputTable.at[index, 'X2_t']
        x_t = cp.array([inputTable.at[index, 'X1_t'],inputTable.at[index, 'X2_t']])
             
        
        delta = inputTable.at[index, 'delta']
        delta_t = inputTable.at[index, 'delta_t'] if 'delta_t' in inputTable.columns else np.nan
        
        if 'd' in inputTable.columns:
            d = inputTable.at[index, 'd']
            d_t, dcof = inputTable.at[index, 'd_t'] if 'd_t' in inputTable.columns else np.nan , np.nan
        else:
            dcof = inputTable.at[index, 'dcof']
            # d_dict = {'none': 0, 'weak': 0.1 * (1 - r), 'strong': 0.6 * (1 - r)}
            d, d_t = dcof*(1-r), np.nan
        #delta_dict = {'none': 0.25, 'weak': 0.15, 'strong': 0.2}
        #delta = delta_dict[EFFECT]

        # d = dcof*(1-r)


        Sigma1_coeff = inputTable.at[index, 'Sigma1_coeff']
        Sigma1_coeff_k = inputTable.at[index, 'Sigma1_coeff_k'] if 'Sigma1_coeff_k' in inputTable.columns else np.nan
        
        x_t_np = x_t.get()  # Convert CuPy array to NumPy array
        print(f'\nRow {index} from {INPUT_PATH}')
        print(f'   t={t}, r={r}, I(s)={Is}, p₁={p_1}, 𝛅={delta}, d={d}, effect:{EFFECT}')
        

        #####################################################################
        whitespace = "        "
        xt = x_t        
        i_degenerate = 1 if r == 1.0 else 0
        deg_type = 'whole.' if r == 1.0 else 'subg. '
        Z_r = cp.array(Z[1-i_degenerate])
        Xt_r = cp.array(xt[1-i_degenerate])

            
        ####################################################################
        # Calculate parameters for origional case
        bsampler = BayesSampler(t=t, r=r, Is=Is, p_1=p_1, delta=delta, d=d, # Sigma_1=Sigma1,
                                    Sigma1_coeff=Sigma1_coeff, random_seed=SEED)
        Sigma_0 = bsampler.Sigma_0

        # Calculate parameters for degenerate case (subgroup | whole population)
        bsampler2 = BayesSampler(t=t, r=r, Is=Is, p_1=p_1, delta=delta, d=d, # Sigma_1=Sigma1,
                                Sigma1_coeff=Sigma1_coeff, degenerate=i_degenerate, random_seed=SEED)
        Sigma_02 = bsampler2.Sigma_0
        

        print(f"\n    -------ITERATION Xₜ={x_t} (Q{np.round(norm.cdf(x_t_np),3)}), @ iteration {TEST_ITERATION}")
        mu_2, mu_2p, p_2 = bsampler.get_posteriorVar(xt)  # !!! cond funct. based on r
        mu_3, mu_3p, sigma2, sigma3 = bsampler.mu_3, bsampler.mu_3p, bsampler.Sigma_2.get(), bsampler.Sigma_3.get()
        mu_22, mu_2p2, p_22 = bsampler2.get_posteriorVar(Xt_r)  
        mu_32, mu_3p2, sigma22, sigma32 = bsampler2.mu_3, bsampler2.mu_3p, bsampler2.Sigma_2, bsampler2.Sigma_3

        print(whitespace, f"𝛍₃={mu_3},𝛍₃ʹ={mu_3p},Σ₃={bsampler.Sigma_3.get()[0]}")
        print(whitespace, f"𝛍₂={mu_2}, 𝛍₂ʹ={mu_2p}, p₂={p_2}, Σ₂={sigma2.reshape(1, -1)},Σ₀={Sigma_0.get()[0]},Σ₁={bsampler.Sigma_1.get()[0]}")
        print(deg_type, f"𝛍₃={mu_32},𝛍₃ʹ={mu_3p2},Σ₃={bsampler2.Sigma_3}")
        print(deg_type, f"𝛍₂={mu_22}, 𝛍₂ʹ={mu_22}, p₂={p_22}, Σ₂={sigma22},Σ₀={Sigma_02},Σ₁ = {bsampler2.Sigma_1}")

        # Sample Delta_Xt
        sstart_time = time.time() #### start counting total running time
        start_time = time.time()
        # Original case
        Delta_Xt = bsampler.sample_Delta_posterior(xt, N_DELTAXT)
        # Sample Delta_Xt for degenerate case (subgroup | whole population)
        Delta_Xt2 = bsampler2.sample_Delta_posterior(Xt_r, N_DELTAXT).reshape(-1, 1)
        end_time = time.time()
        print(f'Sampling Δ|Xₜ took {(end_time - start_time):.4f}s')
        
        ####################################################################
        # Calculate power
        start_time = time.time()
        
        # Calculate power for original case
        pres = power(Z=Z, Xt=xt, t=t, r=r, Is=Is,
                     Sigma_0=Sigma_0, Delta_Xt=Delta_Xt,
                     Nmax=NMAX, lattice_points_col=lattice_points_col,
                     slice_size=SLICE_SIZE)
        # Calculate power for degenerate case (subgroup | whole population)
        pres2 = power_1d(Z=Z_r, Xt=Xt_r, t=t, r=r, Is=Is,
                         Sigma_0=Sigma_02, Delta_Xt=Delta_Xt2, degenerate=i_degenerate,
                         Nmax=NMAX, lattice_points_col=lattice_points_col,
                         slice_size=SLICE_SIZE)
        # print('Check data type',pres2.dtype) # float      
        end_time = time.time()
        # print(f'       Power @ {Z} took {(end_time - start_time):.4f} s')
        # print(f'       Q(Xₜ @ {Z}) = {1 - pres}')
        # print(deg_type,f'Q(Xₜ @ {Z_r}) = {1 - pres2}')

        # Calculate constraint
        print('---------------------constraint')
        start_time = time.time()
        # Constraint for original case
        cres = constraint(Z=Z, Xt=xt, t=t, Sigma_0=Sigma_0, alpha=ALPHA,
                          Nmax=NMAX, lattice_points=lattice_points)
        # Constraint for degenerate case (subgroup | whole population)
        cres2 = constraint_1d(Z=Z_r, Xt=Xt_r, t=t, r=r, alpha=ALPHA, degenerate=i_degenerate,
                              Sigma_0=Sigma_02, Nmax=NMAX, lattice_points=lattice_points)
        end_time = time.time()
        # print(f'       Constraint @ {Z} and @ {Z_r} took {(end_time - start_time):.4f} s')
        # print(f'       alpha(Xₜ @ {Z}) = {ALPHA - cres}')
        # print(deg_type,f'lpha(Xₜ @ {Z_r}) = {ALPHA - cres2}')
        
        ####################################################################
        # SLSQP Minimization
        print('---------------------SLSQP')
        # SLSQP minimizer for original case
        sqp_minimizer = ZSQP(
            power=lambda Z: np.array([
                power(Z=cp.array(Z), Xt=xt, t=t, r=r, Is=Is,
                      Sigma_0=Sigma_0, Delta_Xt=Delta_Xt,
                      Nmax=NMAX, lattice_points_col=lattice_points_col,
                      slice_size=SLICE_SIZE).get()]),
            constraint=lambda Z: np.array([
                constraint(Z=cp.array(Z), Xt=xt, t=t, alpha=ALPHA,
                           Sigma_0=Sigma_0, Nmax=NMAX, lattice_points=lattice_points).get()]))
        # SLSQP minimizer for degenerate case (subgroup | whole population)
        sqp_minimizer2 = ZSQP_1d(
            power=lambda Z_r: np.array([
                power_1d(Z=cp.array(Z_r), Xt=Xt_r, t=t, r=r, Is=Is,
                         Sigma_0=Sigma_02, Delta_Xt=Delta_Xt2, degenerate=i_degenerate,
                         Nmax=NMAX, lattice_points_col=lattice_points_col,
                         slice_size=SLICE_SIZE)]),
            constraint=lambda Z_r: np.array([
                constraint_1d(Z=cp.array(Z_r), Xt=Xt_r, t=t, r=r, alpha=ALPHA, degenerate=i_degenerate,
                              Sigma_0=Sigma_02, Nmax=NMAX, lattice_points=lattice_points)]))
        start_time = time.time()
        sqp_res = sqp_minimizer.minimize(x0=Z.get())
        sqp_res2 = sqp_minimizer2.minimize(x0=cp.asnumpy(Z_r))

        end_time = time.time()
        # Calculate final power 
        final_power = sqp_res['fun']
        final_Z = sqp_res['x']
        final_power2 = sqp_res2['fun']
        final_Z2 = sqp_res2['x']
        print(f'SLSQP @ {final_Z} and @ {final_Z2} took {(end_time - start_time):.4f} s')
        print(f'       SLSQP results are:\n {str(sqp_res)}')
        print(deg_type,f'SLSQP  results are:\n {str(sqp_res2)}') 
        
        print('---------------------constraint after SLSQP')
        start_time = time.time()
        # Final constraint for original case
        final_cres = constraint(Z=cp.array(final_Z), Xt=xt, t=t, Sigma_0=Sigma_0, alpha=ALPHA,
                          Nmax=NMAX, lattice_points=lattice_points)
        # Final constraint for degenerate case (subgroup | whole population)
        final_cres2 = constraint_1d(Z=final_Z2.item(), Xt=Xt_r, t=t, r=r, alpha=ALPHA, degenerate=i_degenerate,
                              Sigma_0=Sigma_02, Nmax=NMAX, lattice_points=lattice_points)
        end_time = time.time()
        print(f'       Constraint @ {final_Z} and @ {final_Z2} took {(end_time - start_time):.4f} s')
        print(f'       alpha(Xₜ @ {final_Z}) = {ALPHA - final_cres}')
        print(deg_type,f'alpha(Xₜ @ {final_Z2}) = {ALPHA - final_cres2}')
        

        eend_time = time.time()
        # Store results in a dictionary
        results = {'idx': index,
            't': t, 'r': r, 'Is': Is, 'p_1': p_1,
            'delta': delta,'delta_t': delta_t, 'd': d, 'd_t': d_t, 'dcof': dcof, 'Sigma1_coeff': Sigma1_coeff, 'Sigma1_coeff_k': Sigma1_coeff_k,
            'X1_t_orig': x1_orig,'X1_t': x1,'X2_t': x2, 'p_2':p_2, 'mu_2': mu_2, 'mu_2p': mu_2p, 'mu_3p': mu_3p, 'Sigma2': sigma2, 'Sigma3': sigma3,
            'p_22':p_22, 'mu_22': mu_22, 'mu_2p2': mu_2p2, 'mu_3p2': mu_3p2, 'Sigma22': sigma22, 'Sigma32': sigma32,
            'solver': 'SLSQP', 'success': sqp_res['success'], 'deg.success': sqp_res2['success'],
            'runtime': eend_time - sstart_time, 'z': final_Z, 'deg.z': final_Z2,
            'power': 1 - final_power, 'deg.power': 1 - final_power2,
            'alpha': ALPHA - final_cres, 'deg.alpha': ALPHA - final_cres2}
        all_results_data.append(results)

        # Check if 2 hours have passed, store values for temp
        current_time = time.time()
        if current_time - total_start_time >= save_interval:
            # Convert the list of results to a DataFrame
            all_results_df = pd.DataFrame(all_results_data,
                                          columns=['idx','t','Is','Sigma1_coeff', 'Sigma1_coeff_k', 'r', 'p_1','delta','delta_t','d','d_t', 'dcof','X1_t_orig','X1_t', 'X2_t','runtime','solver','success', 'deg.success','z', 'power', 'alpha', 'p_2', 'p_22','mu_2', 'mu_2p','mu_22', 'mu_2p2', 'Sigma2','Sigma22', 'mu_3p','mu_3p2', 'Sigma3','Sigma32','deg.z', 'deg.power', 'deg.alpha'])

            # Save the DataFrame to a CSV file
            all_results_df.to_csv(results_csv_path, index=False)

            # Reset the start time
            total_start_time = current_time

    all_results_df = pd.DataFrame(all_results_data, columns=['idx','t','Is','Sigma1_coeff', 'Sigma1_coeff_k', 'r', 'p_1','delta','delta_t','d','d_t', 'dcof','X1_t_orig','X1_t', 'X2_t','runtime','solver','success', 'deg.success','z', 'power', 'alpha', 'p_2', 'p_22','mu_2', 'mu_2p','mu_22', 'mu_2p2', 'Sigma2','Sigma22', 'mu_3p','mu_3p2', 'Sigma3','Sigma32','deg.z', 'deg.power', 'deg.alpha'])


    all_results_df.to_csv(results_csv_path, index=False)
    print(f"Wrote combined results table to {results_csv_path}")
