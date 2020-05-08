# Copyright 2019 Megagon Labs, Inc. and the University of Edinburgh. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import sys

import pandas as pd

from sumeval.metrics.rouge import RougeCalculator
from sumeval.metrics.bleu import BLEUCalculator

from models import SumEvaluator
from utils import Config

if __name__ == "__main__":
    
    if len(sys.argv) < 5:
        print("Config file(s) are missing")
        print("Usage: {} <prepare_conf> <train_conf> <aggregate_conf> <generate_conf>")
        sys.exit(1)

    p_conf = Config(sys.argv[1])
    t_conf = Config(sys.argv[2])
    a_conf = Config(sys.argv[3])
    g_conf = Config(sys.argv[4])
    
    assert p_conf.conf_type == "prepare"
    assert t_conf.conf_type == "train"
    assert a_conf.conf_type == "aggregate"
    assert g_conf.conf_type == "generate"
    
    # Basepath
    if "BASEPATH" not in os.environ:
        basepath = "."
    else:
        basepath = os.environ["BASEPATH"]

    # TODO(Yoshi): YELP dataset hard coded
    data_dirpath = os.path.join(basepath,
                                "data",
                                "{}".format(p_conf.conf_name))
    agg_test_filepath = os.path.join(data_dirpath,
                                     "test{}.csv".format(a_conf.get_agg_name()))

    # output/yelp-default_op2text_small_beam_agg.csv
    agg_pred_filepath = os.path.join(basepath,
                                     "output",
                                     "{}_op2text_{}_{}_{}_agg.csv".format(p_conf.conf_name,
                                                                          t_conf.conf_name,
                                                                          a_conf.conf_name,
                                                                          g_conf.conf_name))

    true_df = pd.read_csv(agg_test_filepath)
    pred_df = pd.read_csv(agg_pred_filepath)

    # if gold_summary is missing, take it from another file
    if "gold_summary" not in true_df:
        print("WARNING: Missing gold_summary. Borrow it from another file.")
        ref_df = pd.read_csv(os.path.join(data_dirpath,
                                          "test_8_10_all_all_300_6.csv"))
        true_df = pd.merge(true_df, ref_df[["eid", "gold_summary"]])
    
    merge_df = pd.merge(true_df[["eid", "gold_summary", "input_text"]], pred_df)
    
    # sumeval evaluator
    evaluator = SumEvaluator(metrics=t_conf["metrics"],
                             stopwords=False,
                             lang="en")

    # Generation evaluation
    eval_df = evaluator.eval(merge_df["gold_summary"].tolist(),
                             merge_df["pred"].tolist())

    eval_df.mean(axis=0).to_csv(agg_pred_filepath.replace(".csv", "_eval.csv"),
                                index=False)
    eval_df.mean(axis=0).to_csv(agg_pred_filepath.replace(".csv", ".eval"))
    eval_df.to_csv(agg_pred_filepath.replace(".csv", "_all.csv"))
    
    pd.concat([merge_df[["eid", "input_text"]], eval_df], axis=1).to_csv(agg_pred_filepath.replace(".csv", "_eval_input.csv"),
                                                                         index=False)
    
