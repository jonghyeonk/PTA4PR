'''
    This file is part of PM4Py (More Info: https://pm4py.fit.fraunhofer.de).

    PM4Py is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PM4Py is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PM4Py.  If not, see <https://www.gnu.org/licenses/>.
'''
import difflib
from enum import Enum
from typing import Optional, Dict, Any, List, Set, Union
import time

from pm4py.objects.log.obj import EventLog, Trace
from pm4py.objects.log.util import log_regex
from pm4py.objects.petri_net.utils import align_utils
from pm4py.util import exec_utils
from pm4py.util import string_distance
from pm4py.util import typing
from pm4py.objects.conversion.log import converter as log_converter

from sklearn.neighbors import NearestNeighbors
import numpy as np
import stringdist
import pandas as pd

class Parameters(Enum):
    PERFORM_ANTI_ALIGNMENT = "perform_anti_alignment"


def apply(log1: EventLog, log2: EventLog, parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> typing.ListAlignments:
    """
    Aligns each trace of the first log against the second log, minimizing the edit distance

    Parameters
    --------------
    log1
        First log
    log2
        Second log
    parameters
        Parameters of the algorithm

    Returns
    ---------------
    aligned_traces
        List that contains, for each trace of the first log, the corresponding alignment
    """
    if parameters is None:
        parameters = {}

    log1 = log_converter.apply(log1, variant=log_converter.Variants.TO_EVENT_LOG, parameters=parameters)
    log2 = log_converter.apply(log2, variant=log_converter.Variants.TO_EVENT_LOG, parameters=parameters)

    anti_alignment = exec_utils.get_param_value(Parameters.PERFORM_ANTI_ALIGNMENT, parameters, False)

    aligned_traces = []

    # form a mapping dictionary associating each activity of the two logs to an ASCII character
    mapping = log_regex.form_encoding_dictio_from_two_logs(log1, log2, parameters=parameters)
    # encode the second log (against which we want to align each trace of the first log)
    list_encodings = log_regex.get_encoded_log(log2, mapping, parameters=parameters)
    # optimization: keep one item per variant
    set_encodings = set(list_encodings)
    list_encodings = list(set_encodings)
    # this initial sort helps in reducing the execution time in the following phases,
    # since the expense of all the successive sorts is reduced
    if anti_alignment:
        list_encodings = sorted(list_encodings, key=lambda x: -len(x))
    else:
        list_encodings = sorted(list_encodings, key=lambda x: len(x))

    # keeps an alignment cache (to avoid re-calculating the same edit distances :) )
    cache_align = {}

    best_worst_cost = min(len(x) for x in list_encodings)

    for trace in log1:
        # gets the alignment
        align_result = align_trace(trace, list_encodings, set_encodings, mapping, cache_align=cache_align,
                                   parameters=parameters)
        aligned_traces.append(align_result)

    # assign fitness to traces
    for index, align in enumerate(aligned_traces):
        if align is not None:
            unfitness_upper_part = align['cost'] // align_utils.STD_MODEL_LOG_MOVE_COST
            if unfitness_upper_part == 0:
                align['fitness'] = 1
            elif (len(log1[index]) + best_worst_cost) > 0:
                align['fitness'] = 1 - (
                        (align['cost'] // align_utils.STD_MODEL_LOG_MOVE_COST) / (len(log1[index]) + best_worst_cost))
            else:
                align['fitness'] = 0
            align["bwc"] = (len(log1[index]) + best_worst_cost) * align_utils.STD_MODEL_LOG_MOVE_COST

    return aligned_traces


def align_trace(trace: Trace, list_encodings: List[str], set_encodings: Set[str], mapping: Dict[str, str],
                cache_align: Optional[Dict[Any, Any]] = None,
                parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> typing.AlignmentResult:
    """
    Aligns a trace against a list of traces, minimizing the edit distance

    Parameters
    --------------
    trace
        Trace
    list_encodings
        List of encoded traces (the same as set_encodings, but as a list)
    set_encodings
        Set of encoded traces (the same as list_encodings, but as a set),
        useful to quickly check if the provided trace is contained in the traces of the other log
    mapping
        Mapping (of activities to characters)
    cache_align
        Cache of the alignments
    parameters
        Parameters of the algorithm

    Returns
    --------------
    aligned_trace
        Aligned trace
    """
    if parameters is None:
        parameters = {}

    # keeps an alignment cache (to avoid re-calculating the same edit distances :) )
    if cache_align is None:
        cache_align = {}

    anti_alignment = exec_utils.get_param_value(Parameters.PERFORM_ANTI_ALIGNMENT, parameters, False)
    comparison_function = string_distance.argmax_levenshtein if anti_alignment else string_distance.argmin_levenshtein

    # encode the current trace using the mapping dictionary
    encoded_trace = log_regex.get_encoded_trace(trace, mapping, parameters=parameters)
    inv_mapping = {y: x for x, y in mapping.items()}

    if encoded_trace not in cache_align:
        if not anti_alignment and encoded_trace in set_encodings:
            # the trace is already in the encodings. we don't need to calculate any edit distance
            argmin_dist = encoded_trace
        else:
            # finds the encoded trace of the other log that is at minimal distance
            argmin_dist = comparison_function(encoded_trace, list_encodings)
        seq_match = difflib.SequenceMatcher(None, encoded_trace, argmin_dist).get_matching_blocks()
        i = 0
        j = 0
        align_trace = []
        total_cost = 0
        for el in seq_match:
            while i < el.a:
                align_trace.append((inv_mapping[encoded_trace[i]], ">>"))
                total_cost += align_utils.STD_MODEL_LOG_MOVE_COST
                i = i + 1
            while j < el.b:
                align_trace.append((">>", inv_mapping[argmin_dist[j]]))
                total_cost += align_utils.STD_MODEL_LOG_MOVE_COST
                j = j + 1
            for z in range(el.size):
                align_trace.append((inv_mapping[encoded_trace[i]], inv_mapping[argmin_dist[j]]))
                i = i + 1
                j = j + 1

        align = {"alignment": align_trace, "cost": total_cost}
        # saves the alignment in the cache
        cache_align[encoded_trace] = align
        return align
    else:
        return cache_align[encoded_trace]
    
    

    
    
###############################


def apply_v2(log1: EventLog, log2: EventLog, parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> typing.ListAlignments:
    """
    Aligns each trace of the first log against the second log, minimizing the edit distance

    Parameters
    --------------
    log1
        First log
    log2
        Second log
    parameters
        Parameters of the algorithm

    Returns
    ---------------
    aligned_traces
        List that contains, for each trace of the first log, the corresponding alignment
    """
    if parameters is None:
        parameters = {}
    start_time1 = time.time()

    log1 = log_converter.apply(log1, variant=log_converter.Variants.TO_EVENT_LOG, parameters=parameters)
    log2 = log_converter.apply(log2, variant=log_converter.Variants.TO_EVENT_LOG, parameters=parameters)

    anti_alignment = exec_utils.get_param_value(Parameters.PERFORM_ANTI_ALIGNMENT, parameters, False)
    exe_time = []
    aligned_traces = []
    row_time = []
    # form a mapping dictionary associating each activity of the two logs to an ASCII character
    mapping = log_regex.form_encoding_dictio_from_two_logs(log1, log2, parameters=parameters)
    # encode the second log (against which we want to align each trace of the first log)
    list_encodings = log_regex.get_encoded_log(log2, mapping, parameters=parameters)
    # optimization: keep one item per variant
    set_encodings = set(list_encodings)
    list_encodings = list(set_encodings)
    # this initial sort helps in reducing the execution time in the following phases,
    # since the expense of all the successive sorts is reduced
    if anti_alignment:
        list_encodings = sorted(list_encodings, key=lambda x: -len(x))
    else:
        list_encodings = sorted(list_encodings, key=lambda x: len(x))

    # print("preprocessing part: --- %s seconds ---" % (time.time() - start_time1))
    exe_time.append((time.time() - start_time1) )

    # keeps an alignment cache (to avoid re-calculating the same edit distances :) )
    cache_align = {}

    best_worst_cost = min(len(x) for x in list_encodings)


    start_time = time.time()
    for n in range(0, len(log1)):
        # gets the alignment
        
        start_row = time.time()
        align_result = align_trace_v2(log1[n], list_encodings, set_encodings, mapping, cache_align=cache_align,
                                parameters=parameters)
        end_row = time.time()
        row_time.append((end_row - start_row) )  
        aligned_traces.append(align_result)
    exe_time.append((time.time() - start_time) )


    return aligned_traces, exe_time, row_time



def align_trace_v2(trace: Trace, list_encodings: List[str], set_encodings: Set[str], mapping: Dict[str, str],
                cache_align: Optional[Dict[Any, Any]] = None, 
                parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> typing.AlignmentResult:
    """
    Aligns a trace against a list of traces, minimizing the edit distance
    
    Parameters
    --------------
    trace
        Trace
    list_encodings
        List of encoded traces (the same as set_encodings, but as a list)
    set_encodings
        Set of encoded traces (the same as list_encodings, but as a set),
        useful to quickly check if the provided trace is contained in the traces of the other log
    mapping
        Mapping (of activities to characters)
    cache_align
        Cache of the alignments
    parameters
        Parameters of the algorithm

    Returns
    --------------
    aligned_trace
        Aligned trace
    """
    if parameters is None:
        parameters = {}


    anti_alignment = exec_utils.get_param_value(Parameters.PERFORM_ANTI_ALIGNMENT, parameters, False)
    comparison_function = string_distance.argmax_levenshtein if anti_alignment else string_distance.argmin_levenshtein_v2


    # encode the current trace using the mapping dictionary
    encoded_trace = log_regex.get_encoded_trace(trace, mapping, parameters=parameters)
    inv_mapping = {y: x for x, y in mapping.items()}
    
    if not anti_alignment and encoded_trace in set_encodings:
        # the trace is already in the encodings. we don't need to calculate any edit distance
        # argmin_dist = encoded_trace
        argmin_dists = comparison_function(encoded_trace, list_encodings)
    else:
        # finds the encoded trace of the other log that is at minimal distance
        argmin_dists = comparison_function(encoded_trace, list_encodings)
    
    
    # argmin_dists2 = argmin_dists[0]
    # argmin_dists2 = argmin_dists

    ## JH : argmin_dists[1] 로 sort 시키고 다음 loop 돌려야함  (만약 순위를 lev. dist.로 할경우에 한정임)
    aligns = list()
    for argmin_dist in list_encodings:
        seq_match = difflib.SequenceMatcher(None, encoded_trace, argmin_dist).get_matching_blocks()
        i = 0
        j = 0
        align_trace = []
        model_trace = []
        agent_trace = []
        total_cost = 0
        for el in seq_match:
            while i < el.a:
                align_trace.append((inv_mapping[encoded_trace[i]], ">>"))
                agent_trace.append( inv_mapping[encoded_trace[i]])
                total_cost += align_utils.STD_MODEL_LOG_MOVE_COST
                i = i + 1
            while j < el.b:
                align_trace.append((">>", inv_mapping[argmin_dist[j]]))
                total_cost += align_utils.STD_MODEL_LOG_MOVE_COST
                model_trace.append( inv_mapping[argmin_dist[j]] )
                j = j + 1
            for z in range(el.size):
                align_trace.append((inv_mapping[encoded_trace[i]], inv_mapping[argmin_dist[j]]))
                agent_trace.append( inv_mapping[encoded_trace[i]])   
                model_trace.append( inv_mapping[argmin_dist[j]] )
                i = i + 1
                j = j + 1
        align = {"alignment": align_trace, "cost": total_cost, 
                 "agent_trace":agent_trace, "model_trace":model_trace }
        # saves the alignment in the cache
        cache_align[encoded_trace] = align
        aligns.append(align)
    return aligns




def apply_v2_knn(log1: EventLog, log2: EventLog, k, df_prob, parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> typing.ListAlignments:
    """
    Aligns each trace of the first log against the second log, minimizing the edit distance

    Parameters
    --------------
    log1
        First log
    log2
        Second log
    parameters
        Parameters of the algorithm

    Returns
    ---------------
    aligned_traces
        List that contains, for each trace of the first log, the corresponding alignment
    """
    if parameters is None:
        parameters = {}
    start_time1 = time.time()
    log1 = log_converter.apply(log1, variant=log_converter.Variants.TO_EVENT_LOG, parameters=parameters)
    log2 = log_converter.apply(log2, variant=log_converter.Variants.TO_EVENT_LOG, parameters=parameters)

    anti_alignment = exec_utils.get_param_value(Parameters.PERFORM_ANTI_ALIGNMENT, parameters, False)
    exe_time = []
    aligned_traces = []
    test_save1 = []
    test_save2 = []
    # form a mapping dictionary associating each activity of the two logs to an ASCII character
    mapping = log_regex.form_encoding_dictio_from_two_logs(log1, log2, parameters=parameters)
    # encode the second log (against which we want to align each trace of the first log)
    list_encodings = log_regex.get_encoded_log(log2, mapping, parameters=parameters)
    # optimization: keep one item per variant
    set_encodings = set(list_encodings)
    list_encodings = list(set_encodings)
    # this initial sort helps in reducing the execution time in the following phases,
    # since the expense of all the successive sorts is reduced
    if anti_alignment:
        list_encodings = sorted(list_encodings, key=lambda x: -len(x))
    else:
        list_encodings = sorted(list_encodings, key=lambda x: len(x))

    # keeps an alignment cache (to avoid re-calculating the same edit distances :) )
    cache_align = {}

    best_worst_cost = min(len(x) for x in list_encodings)

    logtrace_encodings = log_regex.get_encoded_log(log1, mapping, parameters=parameters)
    
    texts = list(set( logtrace_encodings + list_encodings ))
    texts_idx = list(range(0,len(texts)))
    
    
    logtrace_encodings_num = list()
    for i in logtrace_encodings:
        logtrace_encodings_num.append(texts_idx[texts.index(i)]   )
        
    list_encodings_num = list()
    for i in list_encodings:
        list_encodings_num.append( texts_idx[texts.index(i)]  )   

    train = pd.DataFrame({'trace': list_encodings_num, 'prob': df_prob.values})
    
    # print("preprocessing part: --- %s seconds ---" % (time.time() - start_time1))
    exe_time.append((time.time() - start_time1) )
    
    def vector_to_text(texts_idx):
        return texts[int(texts_idx)]

    def mydist2(x, y):
        p = train[train['trace'].isin(y)].prob
        if len(p) != 1:
            p =0
        x, y = map(vector_to_text, (x, y))
        return 100 - p/(1+stringdist.levenshtein(x, y)) # for minimizing problem

    def nearest_neighbors(values, all_values, nbr_neighbors):
        nn = NearestNeighbors(n_neighbors = nbr_neighbors, algorithm='ball_tree',
                            metric='pyfunc', 
                            metric_params={"func":mydist2}).fit(all_values)  # algo: 'auto', 'ball_tree', 'kd_tree', 'brute'
        dists, idxs = nn.kneighbors(values)
        return dists, idxs
    ## start: kNN part
    start_time = time.time()
    n_dists, n_idxs = nearest_neighbors(np.array(logtrace_encodings_num).reshape(-1, 1), np.array(list_encodings_num).reshape(-1, 1), 
                                    nbr_neighbors = k) # 20 , len(sn_encoded_train)

    # print("kNN part: --- %s seconds ---" % (time.time() - start_time))
    exe_time.append((time.time() - start_time) )
    ## end: kNN part

    
    start_time = time.time()
    for i in range(len(log1)):
        trace = log1[i]
        list_encodings_knn = [list_encodings[x] for x in n_idxs[i]] 
        # gets the alignment
        align_result = align_trace_v2_knn(trace, list_encodings_knn, set_encodings, mapping, cache_align=cache_align,
                                parameters=parameters)

        aligned_traces.append(align_result)
    
    
    # print("Alignment part: --- %s seconds ---" % (time.time() - start_time))
    test_save1.append(len(log1) )
    exe_time.append((time.time() - start_time) )
    # for h in n_idxs:
    #     trace = log1[list(h)]
    #     # gets the alignment
    #     align_result = align_trace_v2(trace, list_encodings, set_encodings, mapping, cache_align=cache_align,
    #                             parameters=parameters)
    #     aligned_traces.append(align_result)

    return aligned_traces, exe_time, test_save1


def align_trace_v2_knn(trace: Trace, list_encodings: List[str], set_encodings: Set[str], mapping: Dict[str, str],
                cache_align: Optional[Dict[Any, Any]] = None,
                parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> typing.AlignmentResult:
    """
    Aligns a trace against a list of traces, minimizing the edit distance
    
    Parameters
    --------------
    trace
        Trace
    list_encodings
        List of encoded traces (the same as set_encodings, but as a list)
    set_encodings
        Set of encoded traces (the same as list_encodings, but as a set),
        useful to quickly check if the provided trace is contained in the traces of the other log
    mapping
        Mapping (of activities to characters)
    cache_align
        Cache of the alignments
    parameters
        Parameters of the algorithm

    Returns
    --------------
    aligned_trace
        Aligned trace
    """
    if parameters is None:
        parameters = {}

    # keeps an alignment cache (to avoid re-calculating the same edit distances :) )

    anti_alignment = exec_utils.get_param_value(Parameters.PERFORM_ANTI_ALIGNMENT, parameters, False)
    # comparison_function = string_distance.argmax_levenshtein if anti_alignment else string_distance.argmin_levenshtein_v2

    # encode the current trace using the mapping dictionary
    encoded_trace = log_regex.get_encoded_trace(trace, mapping, parameters=parameters)
    inv_mapping = {y: x for x, y in mapping.items()}

    # if not anti_alignment and encoded_trace in set_encodings:
    #     # the trace is already in the encodings. we don't need to calculate any edit distance
    #     # argmin_dist = encoded_trace
    #     argmin_dists = comparison_function(encoded_trace, list_encodings)
    # else:
    #     # finds the encoded trace of the other log that is at minimal distance
    #     argmin_dists = comparison_function(encoded_trace, list_encodings)
    
    # argmin_dists2 = argmin_dists[0]
    aligns = list()
    for argmin_dist in list_encodings:
        seq_match = difflib.SequenceMatcher(None, encoded_trace, argmin_dist).get_matching_blocks()
        i = 0
        j = 0
        align_trace = []
        total_cost = 0
        for el in seq_match:
            while i < el.a:
                align_trace.append((inv_mapping[encoded_trace[i]], ">>"))
                total_cost += align_utils.STD_MODEL_LOG_MOVE_COST
                i = i + 1
            while j < el.b:
                align_trace.append((">>", inv_mapping[argmin_dist[j]]))
                total_cost += align_utils.STD_MODEL_LOG_MOVE_COST
                j = j + 1
            for z in range(el.size):
                align_trace.append((inv_mapping[encoded_trace[i]], inv_mapping[argmin_dist[j]]))
                i = i + 1
                j = j + 1
        align = {"alignment": align_trace, "cost": total_cost}
        # saves the alignment in the cache
        cache_align[encoded_trace] = align
        aligns.append(align)
    return aligns


        
    
######





def apply_v3(log1: EventLog, log2: EventLog, parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> typing.ListAlignments:
    """
    Aligns each trace of the first log against the second log, minimizing the edit distance

    Parameters
    --------------
    log1
        First log
    log2
        Second log
    parameters
        Parameters of the algorithm

    Returns
    ---------------
    aligned_traces
        List that contains, for each trace of the first log, the corresponding alignment
    """
    if parameters is None:
        parameters = {}

    start_time1 = time.time()
    log1 = log_converter.apply(log1, variant=log_converter.Variants.TO_EVENT_LOG, parameters=parameters)
    log2 = log_converter.apply(log2, variant=log_converter.Variants.TO_EVENT_LOG, parameters=parameters)

    anti_alignment = exec_utils.get_param_value(Parameters.PERFORM_ANTI_ALIGNMENT, parameters, False)
    exe_time = []
    aligned_traces = []
    row_time = []
    # form a mapping dictionary associating each activity of the two logs to an ASCII character
    mapping = log_regex.form_encoding_dictio_from_two_logs(log1, log2, parameters=parameters)
    # encode the second log (against which we want to align each trace of the first log)
    list_encodings = log_regex.get_encoded_log(log2, mapping, parameters=parameters)
    # optimization: keep one item per variant
    set_encodings = set(list_encodings)
    list_encodings = list(set_encodings)
    # this initial sort helps in reducing the execution time in the following phases,
    # since the expense of all the successive sorts is reduced
    if anti_alignment:
        list_encodings = sorted(list_encodings, key=lambda x: -len(x))
    else:
        list_encodings = sorted(list_encodings, key=lambda x: len(x))

    # keeps an alignment cache (to avoid re-calculating the same edit distances :) )
    cache_align = {}

    # print("preprocessing part: --- %s seconds ---" % (time.time() - start_time1))
    exe_time.append((time.time() - start_time1) )
    start_time = time.time()
    best_worst_cost = min(len(x) for x in list_encodings)

    start_time = time.time()
    for trace in log1:
        # gets the alignment
        start_row = time.time()
        align_result = align_trace_v3(trace, list_encodings, set_encodings, mapping, cache_align=cache_align,
                                parameters=parameters)
        end_row = time.time()
        row_time.append((end_row - start_row) )  
        aligned_traces.append(align_result)
        
    # print("Alignment part: --- %s seconds ---" % (time.time() - start_time))
    exe_time.append((time.time() - start_time) )
  
    return aligned_traces, exe_time, row_time


def align_trace_v3(trace: Trace, list_encodings: List[str], set_encodings: Set[str], mapping: Dict[str, str],
                cache_align: Optional[Dict[Any, Any]] = None,
                parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> typing.AlignmentResult:
    """
    Aligns a trace against a list of traces, minimizing the edit distance
    
    Parameters
    --------------
    trace
        Trace
    list_encodings
        List of encoded traces (the same as set_encodings, but as a list)
    set_encodings
        Set of encoded traces (the same as list_encodings, but as a set),
        useful to quickly check if the provided trace is contained in the traces of the other log
    mapping
        Mapping (of activities to characters)
    cache_align
        Cache of the alignments
    parameters
        Parameters of the algorithm

    Returns
    --------------
    aligned_trace
        Aligned trace
    """
    if parameters is None:
        parameters = {}


    anti_alignment = exec_utils.get_param_value(Parameters.PERFORM_ANTI_ALIGNMENT, parameters, False)
    comparison_function = string_distance.argmax_levenshtein if anti_alignment else string_distance.argmin_levenshtein_v2

    # encode the current trace using the mapping dictionary
    encoded_trace = log_regex.get_encoded_trace(trace, mapping, parameters=parameters)
    inv_mapping = {y: x for x, y in mapping.items()}

    if not anti_alignment and encoded_trace in set_encodings:
        # the trace is already in the encodings. we don't need to calculate any edit distance
        # argmin_dist = encoded_trace
        argmin_dists = comparison_function(encoded_trace, list_encodings)
    else:
        # finds the encoded trace of the other log that is at minimal distance
        argmin_dists = comparison_function(encoded_trace, list_encodings)
        
    argmin_dists2 = argmin_dists[0]
    aligns = list()
    for argmin_dist in argmin_dists2:
        
        seq_match = difflib.SequenceMatcher(None, encoded_trace, argmin_dist).get_matching_blocks()
        i = 0
        j = 0
        align_trace = []
        model_trace = []
        agent_trace = []
        total_cost = 0
        for el in seq_match:
            while i < el.a:
                align_trace.append((inv_mapping[encoded_trace[i]], ">>"))
                agent_trace.append( inv_mapping[encoded_trace[i]])
                total_cost += 1000000
                i = i + 1
            while j < el.b:
                align_trace.append((">>", inv_mapping[argmin_dist[j]]))
                total_cost += 1
                model_trace.append( inv_mapping[argmin_dist[j]] )
                j = j + 1
            for z in range(el.size):
                align_trace.append((inv_mapping[encoded_trace[i]], inv_mapping[argmin_dist[j]]))
                agent_trace.append( inv_mapping[encoded_trace[i]])   
                model_trace.append( inv_mapping[argmin_dist[j]] )
                i = i + 1
                j = j + 1
        align = {"alignment": align_trace, "cost": total_cost, 
                 "agent_trace":agent_trace, "model_trace":model_trace }
        # saves the alignment in the cache
        cache_align[encoded_trace] = align
        aligns.append(align)
    
    return aligns
    
def align_trace_v4(trace: Trace, list_encodings: List[str], set_encodings: Set[str], mapping: Dict[str, str],
                cache_align: Optional[Dict[Any, Any]] = None,
                parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> typing.AlignmentResult:
    """
    Aligns a trace against a list of traces, minimizing the edit distance

    Parameters
    --------------
    trace
        Trace
    list_encodings
        List of encoded traces (the same as set_encodings, but as a list)
    set_encodings
        Set of encoded traces (the same as list_encodings, but as a set),
        useful to quickly check if the provided trace is contained in the traces of the other log
    mapping
        Mapping (of activities to characters)
    cache_align
        Cache of the alignments
    parameters
        Parameters of the algorithm

    Returns
    --------------
    aligned_trace
        Aligned trace
    """
    if parameters is None:
        parameters = {}

    # keeps an alignment cache (to avoid re-calculating the same edit distances :) )
    if cache_align is None:
        cache_align = {}

    anti_alignment = exec_utils.get_param_value(Parameters.PERFORM_ANTI_ALIGNMENT, parameters, False)
    comparison_function = string_distance.argmax_levenshtein if anti_alignment else string_distance.argmin_levenshtein

    # encode the current trace using the mapping dictionary
    encoded_trace = log_regex.get_encoded_trace(trace, mapping, parameters=parameters)
    inv_mapping = {y: x for x, y in mapping.items()}




    if encoded_trace not in cache_align:
        if not anti_alignment and encoded_trace in set_encodings:
            # the trace is already in the encodings. we don't need to calculate any edit distance
            argmin_dist = encoded_trace
        else:
            # finds the encoded trace of the other log that is at minimal distance
            argmin_dist = comparison_function(encoded_trace, list_encodings)
        seq_match = difflib.SequenceMatcher(None, encoded_trace, argmin_dist).get_matching_blocks()
        i = 0
        j = 0
        align_trace = []
        total_cost = 0
        for el in seq_match:
            while i < el.a:
                align_trace.append((inv_mapping[encoded_trace[i]], ">>"))
                total_cost += 1000000
                i = i + 1
            while j < el.b:
                align_trace.append((">>", inv_mapping[argmin_dist[j]]))
                total_cost += align_utils.STD_MODEL_LOG_MOVE_COST
                j = j + 1
            for z in range(el.size):
                align_trace.append((inv_mapping[encoded_trace[i]], inv_mapping[argmin_dist[j]]))
                i = i + 1
                j = j + 1

        align = {"alignment": align_trace, "cost": total_cost}
        # saves the alignment in the cache
        cache_align[encoded_trace] = align
        return align
    else:
        return cache_align[encoded_trace]