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

import commentjson


class Config:
    def __init__(self,
                 filepath: str) -> None:
        self.gold_name = False
        if ".csv" in filepath:
            # Assume aggregated file
            filename = os.path.basename(filepath)
            if "gold" in filename:
                conf_name = "-".join(filename.split(".")[0].split("_")[2:])
                self.gold_name = True
            else:
                conf_name = "-".join(filename.split(".")[0].split("_")[1:])
            conf_type = "aggregate"
            self.is_csv = True
            self.config = None
        else:
            filename = os.path.basename(filepath)
            names = filename.split("_")
            conf_type = names[0]
            conf_name = ".".join(
                "_".join(names[1:]).split(".")[:-1])
            self.is_csv = False
            self.load_config(filepath)            
        
        if len(conf_name) == 0:
            raise ValueError("Config name cannot be empty: {}".format(filename))
        
        if conf_type in ["prepare",
                         "train",
                         "aggregate",
                         "generate"]:
            self.conf_type = conf_type
            self.conf_name = conf_name
        else:
            raise ValueError("Invalid config file name: {}".format(filename))
        
    def load_config(self,
                    filepath: str) -> None:
        with open(filepath, "r") as fin:
            lines = fin.readlines()
            json_str = "\n".join(lines)
            self.config = commentjson.loads(json_str)

    def get_agg_name(self):
        """ Generate aggregation target file name."""
        assert self.conf_type == "aggregate"
        if self.is_csv:
            if self.gold_name:
                agg_name = "_gold_{}".format(self.conf_name.replace('-', '_'))
            else:
                agg_name = "_{}".format(self.conf_name.replace('-', '_'))
        else:
            agg_name = "_{}_{}_{}_{}_{}".format(self.config["num_review"],
                                                self.config["top_k"],
                                                "all",
                                                self.config["sentiment"],
                                                self.config["embedding"][-3:],
                                                str(int(self.config["threshold"] * 10))
            )            
            """
            agg_name = "_{}_{}_{}_{}_{}".format(self.config["num_review"],
                                                self.config["top_k"],
                                                self.config["attribute"],
                                                self.config["sentiment"],
                                                self.config["embedding"][-3:],
                                                str(int(self.config["threshold"] * 10))
                                                )
            """
        return agg_name
            
    def __getitem__(self,
                    key: str):
        if key not in self.config:
            raise KeyError(key)
        return self.config[key]

    def __contains__(self,
                     key: str):
        return key in self.config
    

if __name__ == "__main__":
    conf = Config("train_test.json")
