from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import Pipeline
from example1_extra import Preprocessor, FeatureExtractor, load_data
from geneticlearn import Gene, Chromosome, Individual, Population, EnvironmentSKL

model = Pipeline([
                  ('preprocessor', Preprocessor(img_size=(250,250))),
                  ('feature_extractor', FeatureExtractor()),
                  ('ridge_classifier', RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), normalize=True, cv=5))
                 ])

# data
X, y = load_data('example1/data')

#%% hyperparameter to optimize

param_grid = {# 'preprocessor__img_size': [(200, 200), (350, 350), (550, 550)],
              'feature_extractor__glc_offsets': [(3, 7), (5, 10), (3, 7, 11)],
              'knn_classifier__n_neighbors': [3, 5, 7, 11]}

# gen1 = Gene(name='preprocessor__img_size',
#             minv=150,
#             maxv=1000,
#             precision=20,
#             encoding='binary')

coop = lambda x: {'feature_extractor__glc_offsets': (x['feature_extractor__glc_offsets_1'], x['feature_extractor__glc_offsets_2'])}

gen1 = Gene(name='feature_extractor__glc_offsets_1',
            minv=1,
            maxv=30,
            precision=1,
            encoding='binary',
            cooperates='feature_extractor__glc_offsets_2',
            coop_relationship=coop)

gen2 = Gene(name='feature_extractor__glc_offsets_2',
            minv=1,
            maxv=30,
            precision=1,
            encoding='binary',
            coop_relationship=coop)

gen3 = Gene(name='knn_classifier__n_neighbors',
            minv=1,
            maxv=20,
            precision=1,
            encoding='binary')

chr1 = Chromosome(genes=(gen1, gen2, gen3))



























