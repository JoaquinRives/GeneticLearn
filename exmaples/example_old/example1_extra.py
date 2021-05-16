import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import requests
from io import BytesIO
import os
from skimage.feature import greycomatrix, greycoprops
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing


def download_images(file, overwrite=True):
    """
    Download and store images from given URLs

    Parameters:
        filename (str): File containing the URLs

    Returns: None
    """

    filename = os.path.splitext(os.path.basename(file))[0]

    # create directory to store the images
    if not os.path.exists(os.path.join('data', filename)):
        os.makedirs(os.path.join('data', filename))
        print(f"Created {os.path.join('data', filename)}")

    with open(file, 'r') as f:
        print("Downloading images", end='')
        for i, url in enumerate(f.readlines()):
            print(".", end='')
            if url.strip() == '':  # Ignore empty lines
                continue

            response = requests.get(url)
            img = Image.open(BytesIO(response.content))

            ext = img.format
            img_name = f"{filename}_{i}.{ext.lower()}"

            img.save(os.path.join('data', filename, img_name))
    print("DONE \n")


def read_images(data_dir, ext='jpeg'):
    """
    Load images from a given directory.

    Parameters:
        data_dir (str): Directory containing the images

    Returns:
        List of images
    """

    images = []
    for root, dirs, files in os.walk(data_dir):

        for file in files:
            if file.endswith(ext.lower()):
                img = Image.open(os.path.join(root, file))
                images.append(np.array(img))
    return images


def load_data(data_dir, ext='jpeg'):
    """
    Load the image data in a organized dictionary.

    Parameters:
        data_dir (str): Directory containing the images.
        ext (str): Extension of the images (default='jpeg').

    Returns:
        Dictionary of images.
    """

    all_images = []
    all_labels = []
    for root, directories, _ in os.walk(data_dir):
        for directory in directories:
            dir_path = os.path.join(root, directory)
            images = read_images(dir_path, ext='jpeg')
            for img in images:
                all_images.append(img)
                all_labels.append(directory)

    return all_images, np.vstack(all_labels)


def crop_and_resize(images, size=(500, 500)):
    """

    """

    new_images = []
    for img in images:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        # resize and crop keeping the aspect ratio using Lanczos anti-aliasing for the resampling in the resizing.
        new_img = ImageOps.fit(img, size, Image.ANTIALIAS)
        new_images.append(new_img)

    return new_images


def color2gray(images, quantization=False, levels=8):
    """

    """

    new_images = []

    for img in images:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        # resize and crop keeping the aspect ratio using Lanczos anti-aliasing for the resampling in the resizing.
        new_img = img.convert('L')

        if quantization:
            new_img = new_img.quantize(8)

        new_images.append(new_img)

    return new_images


#### DATA PREPARATION FUNCTIONS & CLASSES ####


class GLRLCalculator:
    """
    TODO
    """

    def __init__(self, img_array):

        img_array = img_array.copy()
        if not isinstance(img_array, np.ndarray):
            img_array = np.array(img_array)

        self.img_array = img_array + 1
        self.GRLM_v = None  # Gray-Level-Run-Length matrix (vertical)
        self.GRLM_h = None  # Gray-Level-Run-Length matrix (horizontal)
        self.SRE_h = None  # Short-Run emphasis (horizontal)
        self.SRE_v = None  # Short-Run emphasis (vertical)
        self.LRE_h = None  # Long-Run emphasis (horizontal)
        self.LRE_v = None  # Long-Run emphasis (vertical)
        self.RP_h = None  # Run percentage (horizontal)
        self.RP_v = None  # Run percentage (vertical)

    def calculate_features(self):
        self.calculate_matrix_features(axis=0)
        self.calculate_matrix_features(axis=1)
        return self

    def get_grl_matrix(self, axis=0):
        return self.GRLM_v if axis == 1 else self.GRLM_h

    def get_features(self):
        return self.SRE_h, self.LRE_h, self.RP_h, self.SRE_v, self.LRE_v, self.RP_v

    def calculate_matrix_features(self, axis=0):

        if axis == 1:
            img_array = np.transpose(self.img_array)
        else:
            img_array = self.img_array

        dist1 = 1
        mx = np.max(img_array) + 1
        mn = 0
        gl = (mx - mn)
        p, q = img_array.shape
        n = p * q
        count = 0
        c = 0
        col = 0
        grl = np.zeros((mx, p))
        maxcount = np.zeros((p * q, 1))
        mc = 0

        # compute Gray-Level-Run-Length-Matrix
        for j in range(p):
            for k in range(q - dist1):
                mc = mc + 1
                g = img_array[j, k]
                f = img_array[j, k + dist1]
                if g == f and g != 0:
                    count = count + 1
                    c = count
                    col = count
                    maxcount[mc] = count
                else:
                    grl[g, c] += 1
                    col = 0
                    count = 0
                    c = 0

            grl[f, col] += 1
            count = 0
            c = 0

        GRLM = grl[1:]
        m = GRLM[:, :]
        m1 = np.transpose(m)
        maxrun = int(np.max(maxcount)) + 1
        S = 0
        G = np.zeros((gl, 1))
        R = np.zeros((q, 1))

        for u in range(gl - 1):
            for v in range(q):
                G[u] = G[u] + m[u, v]
                S = S + m[u, v]

        for u1 in range(q):
            for v1 in range(gl - 1):
                R[u1] = R[u1] + m1[u1, v1]

        SRE = 0
        LRE = 0
        RP = 0

        for h1 in range(1, maxrun + 1):
            SRE = SRE + (R[h1 - 1] / (h1 * h1))
            LRE = LRE + (R[h1 - 1] * (h1 * h1))
            RP = RP + R[h1 - 1]

        SRE = SRE / S
        LRE = LRE / S
        RP = RP / n

        if axis == 1:
            self.GRLM_v = GRLM
            self.SRE_v = SRE[0]
            self.LRE_v = LRE[0]
            self.RP_v = RP[0]
        else:
            self.GRLM_h = GRLM
            self.SRE_h = SRE[0]
            self.LRE_h = LRE[0]
            self.RP_h = RP[0]


class Preprocessor(BaseEstimator, TransformerMixin):
    """
    #TODO

    Explain that it inherits from the sklearn classes BaseEstimator and TransformerMixin,
    and implements the sklearn fit/transform methods, which makes it compatible with most
    sklearn pipelines.
    """

    def __init__(self, img_size=500):
        self.img_size = (img_size, img_size)

    def fit(self, X, y=None):
        """ Fit statement, just to accomodate the sklearn pipeline. """
        return self

    def transform(self, X):
        X = X.copy()

        processed_images = []
        for img in X:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)

            proc_img = ImageOps.fit(img, self.img_size, Image.ANTIALIAS)
            processed_images.append(np.array(proc_img))

        return np.array(processed_images)


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    #TODO
    """

    def __init__(self, quantization_lvl=8, glc_offsets_1=3,  glc_offsets_2=7, multiprocess=False, n_jobs=None):
        self.images = None
        self.quantization_lvl = quantization_lvl
        self.features = None
        self.glc_offsets = (glc_offsets_1, glc_offsets_2)
        self.multiprocess = multiprocess

        if self.multiprocess:
            if not n_jobs or n_jobs == -1:
                self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = n_jobs

    def get_features(self, format='df'):
        if format == 'df':
            return pd.DataFrame(self.features)
        else:
            return self.features  # returns the features as a dicctionary

    def fit(self, X, y=None):
        """ Fit statement, just to accomodate the sklearn pipeline """
        return self

    def transform(self, X):
        assert isinstance(X[0], np.ndarray), \
            f"Incorrect image format '{type(X[0])}', the valid format is 'np.ndarray'"

        self.images = X.copy()
        self.features = [{} for i in range(len(self.images))]

        self.calculate_features()

        return self.get_features(format='df')

    def calculate_features(self):
        print("Calculating features...")

        if self.multiprocess:
            print(f"Using multiprocessing (number of CPUs: {self.n_jobs}/{multiprocessing.cpu_count()})")

            Parallel(n_jobs=self.n_jobs)(
                delayed(
                    self.calculate_features_parallel)(i) for i in tqdm(range(len(self.images)))
            )

        else:
            # First level features
            self.first_level_features()
            # Second level features
            self.second_level_features()

    def calculate_features_parallel(self, i):
        # First level features
        self.first_level_features_parallel(i)
        # Second level features
        self.second_level_features_parallel(i)

    def first_level_features(self):
        for i in range(len(self.images)):
            self.features[i].update(self.color_features(self.images[i]))

    def second_level_features(self):
        for i in tqdm(range(len(self.images))):
            assert isinstance(self.images[i], np.ndarray), \
                f"Incorrect image format '{type(img)}', the valid format is 'np.ndarray'"

            img = Image.fromarray(self.images[i])
            img_gray = img.convert('L').quantize(self.quantization_lvl)
            img_gray = np.array(img_gray)

            # GLC features
            self.features[i].update(self.GLC_features(img_gray))
            # GLRL features
            self.features[i].update(self.GLRL_features(img_gray))

    def first_level_features_parallel(self, i):
        self.features[i].update(self.color_features(self.images[i]))

    def second_level_features_parallel(self, i):

        assert isinstance(self.images[i], np.ndarray), \
            f"Incorrect image format '{type(self.images[i])}', the valid format is 'np.ndarray'"

        img = Image.fromarray(self.images[i])
        img_gray = img.convert('L').quantize(self.quantization_lvl)
        img_gray = np.array(img_gray)

        # GLC features
        self.features[i].update(self.GLC_features(img_gray))
        # GLRL features
        self.features[i].update(self.GLRL_features(img_gray))

    def color_features(self, img):
        means = list(img.mean(axis=0).mean(axis=0))
        variances = list(img.var(axis=0).var(axis=0))

        return {'mean_ch1': means[0], 'mean_ch2': means[1], 'mean_ch3': means[2],
                'var_ch1': variances[0], 'var_ch2': variances[1], 'var_ch3': variances[2]}

    def GLC_features(self, img_gray):
        angles = [0, np.pi / 2]
        distances = self.glc_offsets

        glcm = greycomatrix(img_gray, distances=distances, angles=angles,
                            levels=8, symmetric=True, normed=True)

        glc_features = np.hstack(greycoprops(glcm, 'correlation'))

        return {f"glc_{i}": glc_features[i] for i in range(len(glc_features))}

    def GLRL_features(self, img_gray):
        calculator = GLRLCalculator(img_gray)
        glrl_features = calculator.calculate_features().get_features()

        return {f"glrl_{i}": glrl_features[i] for i in range(len(glrl_features))}
