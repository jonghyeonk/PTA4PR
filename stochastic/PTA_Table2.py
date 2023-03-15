# external library
import pm4py
import pandas as pd
import numpy as np
from itertools import chain
import time
import random
import os
org_dir = os.getcwd()
objects = ["activities_64"]
output_path = os.sep.join([str(org_dir), "outputs_PTA"])

# local library
from pm4py.algo.conformance.alignments.edit_distance import algorithm as logs_alignments



for obts in objects:
    
    obj_dir = os.sep.join([str(org_dir),  "datasets", "table_1", obts])
    os.chdir(obj_dir)
    file_list = os.listdir(obj_dir)
    train_list = [s for s in file_list if 'pm4py_' in s]
    goal_list = [s for s in file_list if 'goal' in s]
    
    test = pd.DataFrame({'case:concept:name': [], 'concept:name': [] })

    for i in goal_list:
        goal_path = os.sep.join([str(obj_dir), str(i)])
        obs_list = os.listdir(goal_path)
        os.chdir(goal_path)
        for j in obs_list:
            with open(j) as file:
                obs = [line.rstrip() for line in file]
            caseid= str(i) + '_' +str(j)
            trace = obs[0:(len(obs)-1)]
            set1 = pd.DataFrame()
            set1['concept:name'] = trace
            set1['case:concept:name'] = caseid
            set1['order'] = [*range(0, len(trace))]
            test = pd.concat([test, set1], ignore_index=True)
            
    print('step = stochastic_info')
    os.chdir(obj_dir)
    stochastic_info = list()
    prob_G = []
    for i in range(0,len(train_list)):   # for blocks-world
        log = pm4py.read_xes( 'pm4py_' +str(i) + '.xes')  
        len_log = len(log)
        df3 = pm4py.convert_to_dataframe(log)

        tuple_test2 = df3.groupby('case:concept:name').apply(lambda x: tuple(x['concept:name']))
        df_test2 = pd.DataFrame()
        list_test2 = tuple_test2.to_list()
        df_test2['act'] =list_test2
        df_test2['case'] = tuple_test2.index
        df_test3 = pd.DataFrame({'count' : df_test2.groupby( 'act').size()}).reset_index()

        ref = df_test3['act'].to_list()
        prob = df_test3['count']/sum(df_test3['count'])

        ref_bp = list()

        for trace in ref:
            ref_bp.append( tuple([list(trace)[i] for i,x in enumerate( list(map( lambda x: x.find('tau'), list(trace))) ) if x == -1]) )

        df_bp = pd.DataFrame({'ref_bp':  ref_bp, 'prob' :prob } )
        df_bp = df_bp.groupby(['ref_bp']).apply(lambda x: sum(x['prob']))

        df_bp = pd.DataFrame({'trace' : list(df_bp.index), 'prob': df_bp.values})
        df_bp2 = df_bp.groupby(df_bp['trace']).apply(lambda x: sum(x['prob']))
        
        prob_G.append(len_log)
        stochastic_info.append(df_bp2)

    prob_G = [a/sum(prob_G) for a in prob_G]
    
    
    print('Step: PTA')
    incomplete = [0.3]
    total_result = pd.DataFrame()
    test['order'] = list(chain(*test.groupby('case:concept:name').apply(lambda x: [*range(0,len(x))] ).to_list()))




    for i in incomplete:
        print(i)
        temp = test
        if i == 1.0:
            temp = temp.groupby('case:concept:name', as_index= False).apply(lambda x: x[0:(int(len(x)*i))] ).reset_index()
        else:
            temp = temp.groupby('case:concept:name', as_index= False).apply(lambda x: x[0:(int(len(x)*i)+1)] ).reset_index()
        log_test = temp[['case:concept:name','concept:name']]
        
        list_opt = list()
        list_prob_post = list()
        row_times = []
        for j in stochastic_info:
            df_bp3 = j
            ref = list(df_bp3.index)
            prob = list(df_bp3.values)
                
            caseid = list()
            act = list()
            c = 0
            for k in df_bp3.index:
                act=  act+list(k)
                caseid = caseid + list(np.repeat(c, len(k)))
                c += 1
                
            train_v2 = pd.DataFrame({'case:concept:name': caseid, 'concept:name' : act})

            log2 = pm4py.convert_to_event_log(train_v2)   # size of model traces
            log_test2 = pm4py.convert_to_event_log(log_test)
            
            #Optimal        
            start_time1 = time.time()
            parameters = {}
            see = logs_alignments.apply_v2(log_test2, log2,  parameters=parameters)
            
            start_time2 = time.time()
            
            
            a=0
            see_spn = see[0]
            exe_time = see[1]
            row_time1 = see[2]
            
            row_time2 = []  # For probability calculation
            for h in see_spn:
                                
                p_org = list()
                align =list()
                b=0
                start_row = time.time()
                
                cost_list =[ h[sim]['cost'] for sim in range(0, len(h)) ] 
                beta = 1/(1+min(cost_list))
                bolz_norm = sum( [ np.exp(-1 * beta * cost) for cost in cost_list] )
                for hh in h:
                    
                    act_l = hh['model_trace']
                    incomplete_l = hh['agent_trace']           
                                
                    if tuple(act_l) not in ref:
                        p_org = 0
                        see_spn[a][b]['prob'] = 0
                        see_spn[a][b]['ps'] = 0
                    else:
                        p_org = prob[ ref.index(tuple(act_l)) ]
                        #####
                        prior_obs = [ (str(tuple(incomplete_l))[:-1] in str(r[0:len(incomplete_l)]) and str(tuple(incomplete_l))[:-1] in str(tuple(act_l[0:len(incomplete_l)]))) for r in ref]
                        prior_obs2 = [ (str(tuple(incomplete_l))[:-1] in str(r[0:len(incomplete_l)])) for r in ref]
                        p_ref  = sum([prob[prior_i] for prior_i in range(0,len(prior_obs2)) if prior_obs2[prior_i] == True])
                        p_size = sum([prob[prior_i] for prior_i in range(0,len(prior_obs)) if prior_obs[prior_i] == True])
                        if p_size ==0 :
                            incomplete_l_dec = incomplete_l
                            while p_size ==0  & len(incomplete_l_dec) > 0 :
                                incomplete_l_dec = incomplete_l_dec[:-1]
                                prior_obs = [ (str(tuple(incomplete_l_dec))[:-1] in str(r[0:len(incomplete_l_dec)]) and str(tuple(incomplete_l_dec))[:-1] in str(tuple(act_l[0:len(incomplete_l_dec)]))) for r in ref]
                                prior_obs2 = [ (str(tuple(incomplete_l_dec))[:-1] in str(r[0:len(incomplete_l_dec)])) for r in ref]
                                p_ref  = sum([prob[prior_i] for prior_i in range(0,len(prior_obs2)) if prior_obs2[prior_i] == True])
                            # see_spn[a][b]['prob'] = p_org/p_ref *(len(incomplete_l_dec)/len(incomplete_l))  # prob of model trace  with panelty
                                p_size = len([prob[prior_i] for prior_i in range(0,len(prior_obs)) if prior_obs[prior_i] == True] )
                            if p_size == 0:
                                see_spn[a][b]['prob'] = 0
                                see_spn[a][b]['ps'] = 0
                            else:
                                see_spn[a][b]['prob'] = p_org/(p_ref* (2**(len(incomplete_l)-len(incomplete_l_dec))))
                                see_spn[a][b]['ps'] = p_ref
                        else:
                            see_spn[a][b]['prob'] = p_org/p_ref  # prob of model trace
                            see_spn[a][b]['ps'] = p_ref
                        #####
                                        
                    see_spn[a][b]['similiar'] = (np.exp(-1 * beta * see_spn[a][b]['cost']))/bolz_norm
                    see_spn[a][b]['rank'] = see_spn[a][b]['prob'] * see_spn[a][b]['similiar']  

                    b += 1
                a += 1
                end_row = time.time()
                row_time2.append((end_row - start_row) ) 
            
            ##
            Aligned_trace = list()
            Rank_aligned = list()
            prob_post = list()
            opt_obs = list()
            for n in range(len(see_spn)):
                prob_spn = [see_spn[n][nn]['prob'] for nn in range(len(see_spn[n]))]
                cost_spn = [see_spn[n][nn]['cost'] for nn in range(len(see_spn[n]))]
                rank_spn = [see_spn[n][nn]['rank'] for nn in range(len(see_spn[n]))]
                ps_spn = [see_spn[n][nn]['ps'] for nn in range(len(see_spn[n]))]
                sort_index_spn = np.argsort( rank_spn)[::-1]
                ranked_var_spn = [ tuple(see_spn[n][i]['model_trace']) for i in sort_index_spn]
                #for goal recognition
                Aligned_trace.append(ranked_var_spn[0])
                Rank_aligned.append(1- np.array(rank_spn)[sort_index_spn].tolist()[0])
                prob_post.append(np.array(ps_spn)[sort_index_spn].tolist()[0])
                opt_obs.append(np.array(cost_spn)[sort_index_spn].tolist()[0])
                
                
            res = pd.DataFrame({ 'Trace_org' :  Aligned_trace , 'Rankscore' : Rank_aligned }) 
            
            ##
            
            
            list_prob_post.append(prob_post)
            list_opt.append(opt_obs)
            
            row_time = [x + y for x, y in zip(row_time1, row_time2)]
            if len(row_times)>0:
                row_times = [x + y for x, y in zip(row_times, row_time)]
            else:
                row_times = row_time
            exe_time.append((time.time() - start_time1))   
            

        total_opt = pd.DataFrame()
        total_prob_post = pd.DataFrame()
        for res in range(0, len(list_opt)):
            total_opt['goal'+ str(res)] = list_opt[res]
            total_prob_post['goal'+ str(res)] = list_prob_post[res]
            
        total_prob_post[total_prob_post.apply(sum, axis=1) == 0] = 0.125
        total_prob_post = total_prob_post.apply(lambda row: row /sum( row ), axis= 1)    # This is P(O|G)/P(O)
        
        
        beta = total_opt.apply(lambda row: 1/(1+ np.min(row)) , axis= 1)

        for be in range(0,len(beta)):
            total_opt.loc[be] = np.exp( -1 * beta[be] * total_opt.loc[be]  )
            
        total_opt3 = total_opt.copy()
        
        for be in range(0,len(beta)):
            total_opt.loc[be] = total_prob_post.loc[be] * total_opt.loc[be]  # This is ranking score = P(O|G)/P(O)
            
        total_opt2 = total_opt.apply(lambda row: row /sum( row ), axis= 1)    # This is P(O|G)/P(O)
        total_opt3 = total_opt3.apply(lambda row: row /sum( row ), axis= 1) 
        result = pd.DataFrame()
        result_set = pd.DataFrame()
        for be in range(0,len(total_opt2)):
            result['Prob0'] = [total_opt3.loc[be].values]
            result['Results0'] = random.sample(total_opt3.loc[be].loc[total_opt3.loc[be] ==  max(total_opt3.loc[be])].index.tolist(),1)[0]
            
            total_opt2.loc[be] = prob_G * total_opt2.loc[be]  # This is the objective function P(G|O) = P(G)*P(O|G)/P(O)
            total_opt3.loc[be] = prob_G * total_opt3.loc[be]  # This is the objective function P(G|O) = P(G)*P(O|G)/P(O)
            
            result['Prob1'] = [total_opt3.loc[be].values]
            result['Results1'] = random.sample(total_opt3.loc[be].loc[total_opt3.loc[be] ==  max(total_opt3.loc[be])].index.tolist(),1)[0]
            result['Prob2'] = [total_opt2.loc[be].values]
            result['Results2'] = random.sample(total_opt2.loc[be].loc[total_opt2.loc[be] ==  max(total_opt2.loc[be])].index.tolist(),1)[0]

            result['Incomplete'] = i
            result_set = pd.concat([result_set, result] , ignore_index=True)

        steps = log_test.groupby('case:concept:name').apply(len)
        result_set['Model'] = steps.index
        result_set['Step'] = np.array(steps)
        result_set['Time'] = row_times

        result_set = result_set[['Model', 'Incomplete', 'Step', 'Time', 'Prob0', 'Results0',
                                'Prob1', 'Results1', 'Prob2', 'Results2']]

        total_result = pd.concat([total_result, result_set] , ignore_index=True)

    total_result['Label'] = total_result['Model'].str.split(pat="_sas_").apply(lambda x: x[0])
    os.chdir(output_path)
    total_result.to_csv('table2.csv',  index= False )
    pg = [ 'goal'+str(i) for i in range(0, len(prob_G)) if prob_G[i] == max(prob_G)][0]
    print( "(1) P(G) = " + str(sum(total_result.Label == pg )/len(total_result)))