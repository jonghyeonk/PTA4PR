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
from enum import Enum
from typing import Optional, Dict, Any, Union

import pandas as pd
from pm4py.util import exec_utils, constants, xes_constants


class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY


def apply(df: pd.DataFrame, parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> Dict[str, Dict[str, int]]:
    """
    Computes for each trace of the event log how much rework occurs.
    The rework is computed as the difference between the total number of activities of a trace and the
    number of unique activities.

    Parameters
    ----------------
    df
        Pandas dataframe
    parameters
        Parameters of the algorithm, including:
        - Parameters.ACTIVITY_KEY => the activity key
        - Parameters.CASE_ID_KEY => the case identifier attribute

    Returns
    -----------------
    dict
        Dictionary associating to each case ID:
        - The number of total activities of the case (number of events)
        - The rework (difference between the total number of activities of a trace and the number of unique activities)
    """
    if parameters is None:
        parameters = {}

    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, xes_constants.DEFAULT_NAME_KEY)
    case_id_key = exec_utils.get_param_value(Parameters.CASE_ID_KEY, parameters, constants.CASE_CONCEPT_NAME)

    grouped_df = df.groupby(case_id_key)[activity_key].agg(["count", "nunique"]).reset_index().to_dict("records")
    rework_cases = {}
    for el in grouped_df:
        rework_cases[el["case:concept:name"]] = {"number_activities": el["count"], "rework": el["count"] - el["nunique"]}

    return rework_cases
