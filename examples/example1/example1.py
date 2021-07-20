import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from example1_extra import Preprocessor, FeatureExtractor, load_data
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from geneticlearn import Gene, Chromosome, Individual, Population, EnvironmentSKL
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":

    model = Pipeline([
                      # ('preprocessor', Preprocessor(img_size=(250,250))),
                      # ('feature_extractor', FeatureExtractor(multiprocess=False)),
                      ('normalization', StandardScaler()),
                      ('MLPclassifier', MLPClassifier())
                     ])

    # data
    X, y = load_data('example1/data')
    y = y.ravel()
    X = Preprocessor(img_size=(200, 200)).fit_transform(X)
    X = FeatureExtractor(multiprocess=False).fit_transform(X)

    # #%% Genetic Algorithm


    def relationship(coop_operon):
        phenotype = (coop_operon['MLPclassifier__n_neurons'],) * coop_operon['MLPclassifier__n_layers']
        return {'MLPclassifier__hidden_layer_sizes': phenotype}


    gen1 = Gene(name='MLPclassifier__alpha', encoding='binary', minv=0.000001, maxv=0.01, precision=0.00001)

    gen2 = Gene(name='MLPclassifier__batch_size', encoding='binaryInt', minv=16, maxv=80, precision=1)

    gen3 = Gene(name='MLPclassifier__learning_rate_init', encoding='binary', minv=0.0001, maxv=0.1, precision=0.0005)

    gen4 = Gene(name='MLPclassifier__solver', encoding='categorical', categorical_values=['lbfgs', 'sgd', 'adam'])

    gen5 = Gene(name='MLPclassifier__n_neurons', encoding='binaryInt', minv=1, maxv=100, precision=1,
                cooperates='MLPclassifier__n_layers',
                coop_relationship=relationship)

    gen6 = Gene(name='MLPclassifier__n_layers', encoding='realInt', minv=1, maxv=5, precision=1,
                cooperates='MLPclassifier__n_neurons',
                coop_relationship=relationship)

    gen7 = Gene(name='MLPclassifier__early_stopping', encoding='categorical', categorical_values=[True, False])


    # chromosomes
    chr1 = Chromosome(genes=(gen1, gen2, gen3, gen5), name='chr1', mutation='uniform', recombination='single', r_prob=0.5, m_prob=0.1,
                      r_lambda=1)
    chr2 = Chromosome(genes=(gen4, gen6, gen7), name='chr2', mutation='uniform', recombination='single', r_prob=0.6, m_prob=0.2,
                      r_lambda=1)

    # individual
    individual = Individual(genome=(chr1, chr2), chr_inheritance='independent')

    # population
    population = Population(individual=individual, parallelize=True, parallel_mode='threading')

    # Environment
    metrics = ('f1_macro', 'accuracy')


    def fitness_func(individual):
        acc = individual.scores['mean_test_accuracy'][0]
        f1 = individual.scores['mean_test_f1_macro'][0]
        fitness = (acc + f1) / 2
        print(fitness)
        return fitness


    environment = EnvironmentSKL(model=model, X=X, y=y, metrics=metrics, fitness_func=fitness_func, kfold=3)

    # set the population environment to the new created sklearn environment
    population.environment = environment


    # create first generation
    population.create_individuals(n=100)

    #df = population.evaluate_population()

    history = population.evolve(n_gen=20, selection_method='tournament')


    # #%% Analysis results

    generation_mean = history.groupby('generation').mean()
    generation_std = history.groupby('generation').std()

    history.to_csv('results_ex1.csv')
    generation_mean.to_csv('generation_mean.csv')
    generation_std.to_csv('generation_std.csv')












