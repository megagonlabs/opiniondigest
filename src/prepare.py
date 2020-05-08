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

import csv
import os
import json
import commentjson
from random import shuffle
import re
import sys
from tqdm import tqdm

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk import word_tokenize as tokenizer


class Prepare:
    """Prepare train/dev/test.csv files"""

    def __init__(self, basepath, p_name, source_file, s_min=1, s_max=20, e_min=2, e_max=1000, t_max=150,
                 filter_empty=True, do_stemming=True, num_shuffle=0, split=(0.8, 0.1, 0.1), unk_ratio=-1,
                 word_as_vocab=False):
        self.basepath = basepath
        self.name = p_name
        self.source_file = os.path.join(self.basepath, source_file)
        self.s_min = s_min  # minimum num of sents in a review
        self.s_max = s_max  # maximum num of sents in a review
        self.e_min = e_min  # minimum num of extractions in a review
        self.e_max = e_max  # maximum num of extractions in a review
        self.t_max = t_max  # maximum num of tokens in a review
        self.filter_empty = filter_empty  # filter sent with no extraction
        self.do_stemming = do_stemming
        self.num_shuffle = num_shuffle  # num of shuffles of extractions
        self.split = split  # entity split between train/dev/test
        self.unk_ratio = unk_ratio  # ratio of unknown vocab in test
        self.word_as_vocab = word_as_vocab  # vocab as word (or extraction span)

        self.target_path = os.path.join(self.basepath,
                                        "data",
                                        self.name)
        # create path if not exist
        if not os.path.exists(self.target_path):
            os.makedirs(self.target_path)

        self.stemmer = PorterStemmer() if self.do_stemming else None
        self.detokenizer = TreebankWordDetokenizer()

    def run(self):
        # Read reviews
        print("*****Reading reviews from file*****")
        self.reviews, self.entities = self._read_reviews(self.source_file)

        # Split entities into train/dev/test dataset.
        print("*****Split entities into train/dev/test*****")
        train, dev, test = self._split_train_dev_test()

        # Write to csv files
        print("*****Write files into train/dev/test*****")
        wr_train = csv.writer(open(os.path.join(self.target_path, "train.csv"), "w", encoding="utf-8"))
        wr_dev = csv.writer(open(os.path.join(self.target_path, "dev.csv"), "w", encoding="utf-8"))
        wr_test = csv.writer(open(os.path.join(self.target_path, "test.csv"), "w", encoding="utf-8"))

        # Write header
        wr_train.writerow(["eid", "rid", "review", "extraction", "input_text"])
        wr_dev.writerow(["eid", "rid", "review", "extraction", "input_text"])
        wr_test.writerow(["eid", "rid", "review", "extraction", "input_text"])

        for r_id, review in enumerate(tqdm(self.reviews, desc="reviews")):
            lists = self._review_to_lists(review["ty_id"], r_id, review)
            for row in lists:
                if review["ty_id"] in train:
                    wr_train.writerow(row)
                elif review["ty_id"] in dev:
                    wr_dev.writerow(row)
                else:
                    wr_test.writerow(row)

    def _read_reviews(self, source_file):
        """ Read reviews from file and conduct initial pruning
        """
        entities = set([])
        reviews = []
        num_exts = 0
        with open(source_file, "r", encoding="utf-8") as file:
            for _, line in enumerate(tqdm(file, desc="reviews")):
                review = json.loads(str(line))
                # Process sentences & extractions
                sents = review["sentences"]
                exts = review["extractions"]
                # Filter sentences with NO extractions
                if self.filter_empty:
                    sents = [sents[i] for i in set([e["sid"] for e in exts])]
                # Prune by number of sentences
                if len(sents) < self.s_min or len(sents) > self.s_max:
                    continue
                # Prune by number of extractions
                if len(exts) < self.e_min or len(exts) > self.e_max:
                    continue
                # Process extractions & sentences
                for ext in review["extractions"]:
                    ext["opinion"] = self._process_span(ext["opinion"])
                    ext["aspect"] = self._process_span(ext["aspect"])
                sents = [self.detokenizer.detokenize(toks) for toks in sents]
                # Validate number of tokens per review
                num_tokens = len(tokenizer(" ".join(sents)))
                if num_tokens > self.t_max:
                    continue
                review["sentences"] = sents
                reviews.append(review)
                entities.add(review["ty_id"])
                num_exts += len(exts)
        print("Average number of extractions per review: {}".format(num_exts / (0.0 + len(reviews))))
        return reviews, entities

    def _split_train_dev_test(self):
        """ Split training, validating, and testing dataset.
        """
        train, dev, test = [], [], []
        num_train = int(len(self.entities) * self.split[0])
        num_dev = int(len(self.entities) * self.split[1])
        num_test = len(self.entities) - num_train - num_dev

        if self.unk_ratio < 0:
            entities = list(self.entities)
            shuffle(entities)
            train = entities[:num_train]
            dev = entities[num_train:num_train + num_dev]
            test = entities[num_train + num_dev:]
            return train, dev, test

        # Get statistics
        vocab_size, entity_freq = self._create_stats(self.word_as_vocab)
        test_vocab = int(vocab_size * self.unk_ratio)

        # Select entity in test set in greedy fashion
        entities = list(self.entities)
        shuffle(entities)
        cur_test_vocab = 0
        for e_id in entities:
            freq = entity_freq[e_id]
            if cur_test_vocab + freq <= test_vocab and len(test) <= num_test:
                test.append(e_id)
            else:
                train.append(e_id)

        # Select dev from train set
        dev = train[:num_dev]
        train = train[num_dev:]
        return train, dev, test

    def _extraction2input(self, extraction, sep="[SEP]"):
        return " {} ".format(sep).join([" ".join(e.split(",")[0:2]) for e in extraction.split(";")])

    def _review_to_lists(self, e_id, r_id, review):
        sents = review["sentences"]
        exts = []
        for ext in review["extractions"]:
            opinion = self.detokenizer.detokenize(ext["opinion"])
            aspect = self.detokenizer.detokenize(ext["aspect"])
            ext_item = [opinion, aspect, ext["attribute"], ext["sentiment"]]
            exts.append(",".join(ext_item))
        lists = []
        # lists.append([str(e_id), str(r_id), " ".join(sents), ";".join(exts)])
        lists.append([str(e_id), str(r_id), " ".join(sents), ";".join(exts), self._extraction2input(";".join(exts))])
        for i in range(self.num_shuffle):
            shuffle(exts)
            # lists.append([str(e_id), str(r_id), " ".join(sents), ";".join(exts)])
            lists.append(
                [str(e_id), str(r_id), " ".join(sents), ";".join(exts), self._extraction2input(";".join(exts))])
        return lists

    def _create_stats(self, split_word=False):
        """ Create vocab frequency statistics for entities.
		Args:
			split_word (True or False): when set to True, "vocab" is based on
				extractions; otherwise, "vocab" is based on individual word
		Returns:
			vocab_size (int): total vocabulary size
			entity_low_freq_count (dict): low freq vocab count for each entity
		"""
        # Count vocab frequency: "vocab": frequency & initialize entity freq
        vocab_freq = {}
        entity_freq = {}
        for _, review in enumerate(tqdm(self.reviews, desc="build_vocab")):
            if review["ty_id"] not in entity_freq:
                entity_freq[review["ty_id"]] = 0
            for ext in review["extractions"]:
                for vocab in self._extraction_to_vocab(ext, split_word):
                    if vocab not in vocab_freq:
                        vocab_freq[vocab] = 0
                    vocab_freq[vocab] += 1

        # Update entity low freq count: "entity": low freq vocab count
        for _, review in enumerate(tqdm(self.reviews, desc="update_stats")):
            # Collect vocab for all extractions
            low_freq_vocab = 0
            for ext in review["extractions"]:
                for vocab in self._extraction_to_vocab(ext, split_word):
                    low_freq_vocab += 1 if vocab_freq[vocab] <= 1 else 0
            entity_freq[review["ty_id"]] += low_freq_vocab

        return len(vocab_freq), entity_freq

    def _extraction_to_vocab(self, extraction, split_word):
        """ Create vocabulary for each extraction
		"""
        vocabs = extraction["opinion"] + extraction["aspect"]
        if split_word:
            return [self.detokenizer.detokenize(vocabs)]
        return vocabs

    def _process_span(self, span):
        """ Tokenize a span and stemming tokens if required.
		"""
        span = re.sub(r',', '', span)
        span = re.sub(r';', '', span)
        tokens = nltk.word_tokenize(span)
        if self.do_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        return tokens


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please specify prepare configuration file!"
    config_file = sys.argv[1]
    with open(config_file, "r") as file:
        configs = commentjson.loads(file.read())

    if "BASEPATH" not in os.environ:
        basepath = "."
    else:
        basepath = os.environ["BASEPATH"]

    filter_empty = True if configs["filter_empty"].lower() == "True" else False
    do_stemming = True if configs["do_stemming"].lower() == "True" else False
    word_as_vocab = True if configs["word_as_vocab"].lower() == "True" else False
    preparer = Prepare(basepath,
                       configs["p_name"], configs["source_file"],
                       s_min=configs["s_min"], s_max=configs["s_max"],
                       e_min=configs["e_min"], e_max=configs["e_max"],
                       t_max=configs["t_max"],
                       filter_empty=filter_empty,
                       do_stemming=do_stemming,
                       num_shuffle=configs["num_shuffle"],
                       split=configs["split"],
                       unk_ratio=configs["unk_ratio"],
                       word_as_vocab=word_as_vocab)

    preparer.run()
