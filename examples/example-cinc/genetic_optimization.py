""" Genetic Algorithm (GA) optimization of the model predictions:

It uses GA to obtain weigths for each label ("magic" vector) that maximizes
the total score of the predictions when evaluated with the scoring funtion
of the competition.

Preliminary phase CinC-Challenge-2020 (example with only 9 labels).

@author: Joaquin Rives
@email: joaquin.rives01@gmail.com
@date: March 2021
"""
import copy
import numpy as np
import pandas as pd
import os, os.path, sys
from geneticlearn import Gene, Chromosome, Individual, Population, Environment
from evaluation import evaluate_12ECG_score, compute_beta_score , compute_auc, get_classes, get_true_labels
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    ###################################################################################################################
    # Genetic Algorithm (GA)

    gen1 = Gene(name='w1', encoding='binary', minv=0, maxv=2, length=6)  # AF
    gen2 = Gene(name='w2', encoding='binary', minv=0, maxv=2, length=6)  # I-AVB
    gen3 = Gene(name='w3', encoding='binary', minv=0, maxv=2, length=6)  # LBBB
    gen4 = Gene(name='w4', encoding='binary', minv=0, maxv=2, length=6)  # Normal
    gen5 = Gene(name='w5', encoding='binary', minv=0, maxv=2, length=6)  # PAC
    gen6 = Gene(name='w6', encoding='binary', minv=0, maxv=2, length=6)  # PVC
    gen7 = Gene(name='w7', encoding='binary', minv=0, maxv=2, length=6)  # RBBB
    gen8 = Gene(name='w8', encoding='binary', minv=0, maxv=2, length=6)  # STD
    gen9 = Gene(name='w9', encoding='binary', minv=0, maxv=2, length=6)  # STE

    # chromosomes
    chr1 = Chromosome(genes=(gen1, gen2, gen3, gen4, gen5, gen6, gen7, gen8, gen9), name='chr1', mutation='uniform',
                      recombination='poisson', r_prob=0.5, m_prob=0.1,
                      r_lambda=1)

    # individual
    individual = Individual(genome=(chr1,), chr_inheritance='independent')

    # population
    population = Population(individual=individual, parallelize=True, parallel_mode='threading')


    # Environment
    def fitness_func(X, y, param_grid):

        weights = np.array([v[0] for v in param_grid.values()])

        prob_optimized = weights[None,] * X
        final_prediction = np.where(prob_optimized > 0.5, 1, 0)

        # Compute F_beta measure and the generalization of the Jaccard index
        accuracy, f_measure, Fbeta_measure, Gbeta_measure = compute_beta_score(y, final_prediction, 2, num_classes)

        # compute AUROC and AUPRC
        auroc, auprc = compute_auc(labels, prob_optimized, num_classes)

        scores = {'accuracy': accuracy, 'f_measure': f_measure, 'Fbeta_measure': Fbeta_measure,
                  'Gbeta_measure': Gbeta_measure, 'auroc': auroc, 'auprc': auprc}

        fitness = Fbeta_measure + Gbeta_measure  # + auroc ???

        return fitness, scores

    ###################################################################################################################
    # LABELS = VALIDATION_LABELS
    # TAL, TAL = model.predict(validation)

    # X = output_probabilities
    # y = labels

    environment = Environment(X=X, y=y, fitness_func=fitness_func)
    ###################################################################################################################
    # Evolution

    # set the population environment to the new created sklearn environment
    population.environment = environment

    # create first generation
    population.create_individuals(n=100)


    history = population.evolve(n_gen=5, selection_method='elitism')
    #
    index = history['fitness'].argmax()

    winner = history.iloc[index]
    
    print(f"Winner is: {winner}")

    ###################################################################################################################
    # Test

    weights = winner[['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9']].values

    real_labels = copy.deepcopy(y_test)

    predicted_prob = copy.deepcopy(X_test)
    predicted_labels = np.where(predicted_prob > 0.5, 1, 0)

    optimized_prob = (X_test * weights).astype('float32')
    optimized_labels = np.where(optimized_prob > 0.5, 1, 0)

    # comparison
    accuracy, f_measure, Fbeta_measure, Gbeta_measure = compute_beta_score(real_labels, predicted_labels,
                                                                           2, n_outputs)
    accuracy_opt, f_measure_opt, Fbeta_measure_opt, Gbeta_measure_opt = compute_beta_score(real_labels, optimized_labels,
                                                                                           2, n_outputs)
    # compute AUROC and AUPRC
    auroc, auprc = compute_auc(real_labels, predicted_prob, n_outputs)
    auroc_opt, auprc_opt = compute_auc(real_labels, optimized_prob, n_outputs)

    baseline = [accuracy, f_measure, Fbeta_measure, Gbeta_measure]
    optimization = [accuracy_opt, f_measure_opt, Fbeta_measure_opt, Gbeta_measure_opt]

    results_df = pd.DataFrame([baseline, optimization], columns=['acc', 'f_measure', 'Fbeta', 'Gbeta'])

    print(results_df)

