# -*- coding: utf-8 -*-
#
# Copyright 2019 Pietro Barbiero, Giovanni Squillero and Alberto Tonda
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import random
import copy
from typing import Union, List

import inspyred
import datetime
import numpy as np
import multiprocessing

from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
import warnings
import pandas as pd

warnings.filterwarnings("ignore")


class MECO(BaseEstimator, TransformerMixin):
    """
    EvoFS class.
    """

    def __init__(self, estimator, compression: str = 'both',
                 pop_size: int = 100, max_generations: int = 100,
                 max_features: int = 100, min_features: int = 10,
                 max_samples: int = 500, min_samples: int = 50,
                 n_splits: int = 3, random_state: int = 42,
                 scoring: str = 'f1_weighted', verbose: bool = True,
                 scores: Union[List, np.array] = None, score_func: callable = f_classif):

        self.estimator = estimator
        self.compression = compression
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.max_features = max_features
        self.min_features = min_features
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.n_splits = n_splits
        self.random_state = random_state
        self.scoring = scoring
        self.verbose = verbose
        self.scores = scores
        self.score_func = score_func

    def fit(self, X, y=None, **fit_params):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        n = int(X.shape[0])
        k = int(X.shape[1])

        self.max_generations_ = np.min([self.max_generations, int(math.log10(2 ** int(0.01 * k * n)))])
        self.pop_size_ = np.min([self.pop_size, int(math.log10(2 ** k))])
        self.offspring_size_ = 2 * self.pop_size_
        self.maximize_ = True
        self.individuals_ = []
        self.scorer_ = get_scorer(self.scoring)
        self.max_features_ = np.min([k, self.max_features])
        self.max_samples_ = np.min([n, self.max_samples])

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state)
        list_of_splits = [split for split in skf.split(X, y)]
        trainval_index, test_index = list_of_splits[0]
        self.x_trainval_, x_test = X.iloc[trainval_index], X.iloc[test_index]
        self.y_trainval_, y_test = y[trainval_index], y[test_index]

        self.skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state)
        # list_of_splits2 = [split for split in self.skf.split(self.x_trainval_, self.y_trainval_)]
        # train_index, val_index = list_of_splits2[0]
        # self.x_train_, self.x_val = self.x_trainval_.iloc[train_index], self.x_trainval_.iloc[val_index]
        # self.y_train_, self.y_val = self.y_trainval_[train_index], self.y_trainval_[val_index]

        # rank features
        if self.scores is None:
            fs = SelectKBest(self.score_func, k=1)
            fs.fit(self.x_trainval_, self.y_trainval_)
            self.scores_ = np.nan_to_num(fs.scores_, nan=0)
        else:
            self.scores_ = self.scores

        # initialize pseudo-random number generation
        prng = random.Random()
        prng.seed(self.random_state)

        ea = inspyred.ec.emo.NSGA2(prng)
        ea.variator = [self._variate]
        ea.terminator = inspyred.ec.terminators.generation_termination
        ea.observer = self._observe

        ea.evolve(
            generator=self._generate,

            evaluator=self._evaluate,
            # this part is defined to use multi-process evaluations
            # evaluator=inspyred.ec.evaluators.parallel_evaluation_mp,
            # mp_evaluator=self._evaluate_feature_sets,
            # mp_num_cpus=multiprocessing.cpu_count()-2,

            pop_size=self.pop_size_,
            num_selected=self.offspring_size_,
            maximize=self.maximize_,
            max_generations=self.max_generations_,

            # extra arguments here
            current_time=datetime.datetime.now()
        )

        print('Training completed!')

        # find best individual, the one with the highest accuracy on the validation set
        accuracy_best = 0
        self.solutions_ = []
        feature_counts = np.zeros(X.shape[1])
        for individual in ea.archive:

            feature_set = individual.candidate[1]
            feature_counts[feature_set] += 1

            if self.compression == 'features':
                x_reduced = self.x_trainval_.iloc[:, individual.candidate[1]]
                y_reduced = self.y_trainval_
                x_test_reduced = x_test.iloc[:, individual.candidate[1]]
            elif self.compression == 'samples':
                x_reduced = self.x_trainval_.iloc[individual.candidate[0]]
                y_reduced = self.y_trainval_[individual.candidate[0]]
                x_test_reduced = x_test
            elif self.compression == 'both':
                x_reduced = self.x_trainval_.iloc[individual.candidate[0], individual.candidate[1]]
                y_reduced = self.y_trainval_[individual.candidate[0]]
                x_test_reduced = x_test.iloc[:, individual.candidate[1]]

            model = copy.deepcopy(self.estimator)
            model.fit(x_reduced, y_reduced)

            # compute validation accuracy
            accuracy_test = self.scorer_(model, x_test_reduced, y_test)

            if accuracy_best < accuracy_test:
                self.best_set_ = {
                    'samples': individual.candidate[0],
                    'features': individual.candidate[1],
                    'accuracy': accuracy_test,
                }
                accuracy_best = accuracy_test

            individual.validation_score_ = accuracy_test
            self.solutions_.append(individual)

        self.feature_ranking_ = np.argsort(feature_counts)
        return self

    def transform(self, X, **fit_params):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if self.compression == 'features':
            return X.iloc[:, self.best_set_['features']].values
        elif self.compression == 'samples':
            return X.iloc[self.best_set_['samples']].values
        elif self.compression == 'both':
            return X.iloc[self.best_set_['samples'], self.best_set_['features']].values

    # initial random generation of feature sets
    def _generate(self, random, args):
        individual_f, individual_s = [], []

        if self.compression == 'features' or self.compression == 'both':
            n_features = random.randint(self.min_features, self.max_features_)
            individual_f = np.random.choice(self.x_trainval_.shape[1], size=(n_features,), replace=False).tolist()
            individual_f = np.sort(individual_f).tolist()

        if self.compression == 'samples' or self.compression == 'both':
            n_samples = random.randint(self.min_samples, self.max_samples_)
            individual_s = np.random.choice(self.x_trainval_.shape[0], size=(n_samples,), replace=False).tolist()
            individual_s = np.sort(individual_s).tolist()

        individual = [individual_s, individual_f]

        return individual

    # using inspyred's notation, here is a single operator that performs both crossover and mutation, sequentially
    def _variate(self, random, candidates, args):
        nextgen_f, nextgen_s = [[] for _ in range(len(candidates))], [[] for _ in range(len(candidates))]
        if self.compression == 'features' or self.compression == 'both':
            candidates_f = [c[1] for c in candidates]
            nextgen_f = self._do_variation(random, candidates_f, self.min_features,
                                           self.max_features_, self.x_trainval_.shape[1], args)

        if self.compression == 'samples' or self.compression == 'both':
            candidates_s = [c[0] for c in candidates]
            nextgen_s = self._do_variation(random, candidates_s, self.min_samples,
                                           self.max_samples_, self.x_trainval_.shape[0], args)

        next_generation = [[cs, cf] for cs, cf in zip(nextgen_s, nextgen_f)]
        return next_generation

    def _do_variation(self, random, candidates, min_candidate_size, max_candidate_size, max_size, args):
        split_idx = int(len(candidates) / 2)
        fathers = candidates[:split_idx]
        mothers = candidates[split_idx:]
        next_generation = []
        parent = np.zeros((max_size), dtype=int)

        for father, mother in zip(fathers, mothers):
            parent1 = 0*parent
            parent1[father] = 1
            parent2 = 0*parent
            parent2[mother] = 1

            # well, for starters we just crossover two individuals, then mutate
            children = [list(parent1), list(parent2)]

            # one-point crossover!
            cut_point = random.randint(0, len(children[0]) - 1)
            for index in range(0, cut_point + 1):
                temp = children[0][index]
                children[0][index] = children[1][index]
                children[1][index] = temp

            # mutate!
            for child in children:
                mutation_point = random.randint(0, len(child) - 1)
                if child[mutation_point] == 0:
                    child[mutation_point] = 1
                else:
                    child[mutation_point] = 0

            # check if individual is still valid, and (in case it isn't) repair it
            next_gen = []
            for child in children:
                child = np.array(child)
                points_selected = list(np.argwhere(child == 1).squeeze())
                points_not_selected = list(np.argwhere(child == 0).squeeze())

                if len(points_selected) > max_candidate_size:
                    index = np.random.choice(points_selected, len(points_selected) - max_candidate_size)
                    child[index] = 0

                if len(points_selected) < min_candidate_size:
                    index = np.random.choice(points_not_selected, min_candidate_size - len(points_selected))
                    child[index] = 1

                points_selected = list(np.argwhere(child == 1).squeeze())
                next_gen.append(points_selected)

            next_generation.append(next_gen[0])
            next_generation.append(next_gen[1])

        return next_generation

    # function that evaluates the feature sets
    def _evaluate(self, candidates, args):
        fitness = []
        list_of_splits2 = [split for split in self.skf.split(self.x_trainval_, self.y_trainval_)]
        train_index, val_index = list_of_splits2[np.random.randint(0, self.skf.n_splits)]
        x_train_, x_val = self.x_trainval_.iloc[train_index], self.x_trainval_.iloc[val_index]
        y_train_, y_val = self.y_trainval_[train_index], self.y_trainval_[val_index]
        for c in candidates:
            if self.compression == 'features':
                x_reduced = x_train_.iloc[:, c[1]]
                y_reduced = y_train_
                x_val_reduced = x_val.iloc[:, c[1]]
            elif self.compression == 'samples':
                x_reduced = x_train_.iloc[c[0]]
                y_reduced = y_train_[c[0]]
                x_val_reduced = x_val
            elif self.compression == 'both':
                x_reduced = x_train_.iloc[c[0], c[1]]
                y_reduced = y_train_[c[0]]
                x_val_reduced = x_val.iloc[:, c[1]]

            model = copy.deepcopy(self.estimator)
            # scores = cross_validate(model, x_reduced, y_reduced, scoring=self.scorer_, cv=self.n_splits)
            # cv_scores = np.mean(scores["test_score"])
            model.fit(x_reduced, y_reduced)
            cv_scores = model.score(x_val_reduced, y_val)

            # compute numer of unused features
            samples_removed = x_train_.shape[0] - len(c[0])
            features_removed = x_train_.shape[1] - len(c[1])

            # the best feature sets should contain features which are useful individually
            test_median = np.median(self.scores_[c[1]])

            # maximizing the points removed also means
            # minimizing the number of points taken (LOL)
            objectives = []
            if self.compression == 'samples' or self.compression == 'both':
                objectives.append(samples_removed)
            if self.compression == 'features' or self.compression == 'both':
                objectives.append(features_removed)
            objectives.extend([cv_scores, test_median])

            fitness.append(inspyred.ec.emo.Pareto(objectives))

        return fitness

    # the 'observer' function is called by inspyred algorithms at the end of every generation
    def _observe(self, population, num_generations, num_evaluations, args):
        sample_size = self.x_trainval_.shape[0]
        feature_size = self.x_trainval_.shape[1]
        old_time = args["current_time"]
        # logger = args["logger"]
        current_time = datetime.datetime.now()
        delta_time = current_time - old_time

        # I don't like the 'timedelta' string format,
        # so here is some fancy formatting
        delta_time_string = str(delta_time)[:-7] + "s"

        best_candidate_id = np.argmax(np.array([candidate.fitness[2] for candidate in args['_ec'].archive]))
        # best_candidate_id = np.argmax(np.array([candidate.fitness[2] for candidate in population]))
        best_candidate = args['_ec'].archive[best_candidate_id]
        # best_candidate = population[0]

        log = f"[{delta_time_string}] Generation {num_generations}, Best individual: "
        if self.compression == 'samples' or self.compression == 'both':
            log += f"#samples={len(best_candidate.candidate[0])} (of {sample_size}), "
        if self.compression == 'features' or self.compression == 'both':
            log += f"#features={len(best_candidate.candidate[1])} (of {feature_size}), "
        log += f"accuracy={best_candidate.fitness[-2]*100:.2f}, test={best_candidate.fitness[-1]:.2f}"

        if self.verbose:
            print(log)
        #     logger.info(log)

        args["current_time"] = current_time


# # well, for starters we just crossover two individuals, then mutate
# children = [list(father), list(mother)]
#
# # one-point crossover!
# cut_point1 = random.randint(1, len(children[0])-1)
# cut_point2 = random.randint(1, len(children[1])-1)
# child1 = children[0][cut_point1:] + children[1][:cut_point2]
# child2 = children[1][cut_point2:] + children[0][:cut_point1]
#
# # remove duplicates
# child1 = np.unique(child1).tolist()
# child2 = np.unique(child2).tolist()
# children = [child1, child2]
#
# # mutate!
# for child in children:
#     mutation_point = random.randint(0, len(child)-1)
#     while True:
#         new_val = np.random.choice(max_size)
#         if new_val not in child:
#             child[mutation_point] = new_val
#             break
#
# # check if individual is still valid, and
# # (in case it isn't) repair it
# for child in children:
#
#     # if it has too many features, delete them
#     if len(child) > max_candidate_size:
#         n_surplus = len(child) - max_candidate_size
#         indexes = np.random.choice(len(child), size=(n_surplus,))
#         child = np.delete(child, indexes).tolist()
#
#     # if it has too less features, add more
#     if len(child) < min_candidate_size:
#         n_surplus = min_candidate_size - len(child)
#         for _ in range(n_surplus):
#             while True:
#                 new_val = np.random.choice(max_size)
#                 if new_val not in child:
#                     child.append(new_val)
#                     break
#
# children[0] = np.sort(children[0]).tolist()
# children[1] = np.sort(children[1]).tolist()
# next_generation.append(children[0])
# next_generation.append(children[1])