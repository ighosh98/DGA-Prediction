"""Generates data for train/test algorithms"""
from datetime import datetime
from StringIO import StringIO
from urllib import urlopen
from zipfile import ZipFile

import cPickle as pickle
import os
import random
import tldextract
import numpy as np

from dga_classifier.dga_generators import matsnu, suppobox, gozi,pykspa

# Location of Alexa 1M
ALEXA_1M = 'http://s3.amazonaws.com/alexa-static/top-1m.csv.zip'

DATA_FILE = 'traindata.pkl'

def get_alexa(num, address=ALEXA_1M, filename='top-1m.csv'):
    url = urlopen(address)
    zipfile = ZipFile(StringIO(url.read()))
    return [tldextract.extract(x.split(',')[1]).domain for x in \
            zipfile.read(filename).split()[:num]]

def gen_malicious(num_per_dga=10000):
    """Generates num_per_dga of each DGA"""
    domains = []
    labels = []
    
    # Generate pyskpa domains
    domains += pykspa.generate_domains(num_per_dga, datetime(2016, 1, 1))
    labels += ['pykspa']*num_per_dga
    
    # matsnu
    domains += matsnu.generate_domains(num_per_dga, include_tld=False)
    labels += ['matsnu']*num_per_dga

    # suppobox
    domains += suppobox.generate_domains(num_per_dga, include_tld=False)
    labels += ['suppobox']*num_per_dga

    # gozi
    domains += gozi.generate_domains(num_per_dga, include_tld=False)
    labels += ['gozi']*num_per_dga

    return domains, labels

def gen_data(force=False):
    """Grab all data for train/test and save

    force:If true overwrite, else skip if file
          already exists
    """
    if force or (not os.path.isfile(DATA_FILE)):
        domains, labels = gen_malicious(10000)
        domains += get_alexa(len(domains))
        labels += ['benign']*len(domains)

        pickle.dump(zip(labels, domains), open(DATA_FILE, 'w'))

def get_data(force=False):
    """Returns data and labels"""
    gen_data(force)

    return pickle.load(open(DATA_FILE))

def get_malware_labels(labels):
    malware_labels = list(set(labels))
    malware_labels.remove('benign')
    malware_labels.sort()
    return malware_labels


def expand_labels(labels):
    '''
    This function takes the labels as returned from get_data()
    and it converts them into a list of lists of 0/1 labels per
    'benign' and per each malware family
    '''

    # Convert labels to 0-1
    y = [0 if label == 'benign' else 1 for label in labels]
    all_Ys = [y]
    for malw_label in get_malware_labels(labels):
        all_Ys.append([1 if label == malw_label else 0 for label in labels])
    return all_Ys

def get_labels():
    return [
        'main',
        'matsnu',
        'suppobox',
        'gozi',
        'pykspa',
    ]

def get_losses():
    return {
        'main': 'binary_crossentropy',
        'matsnu': 'binary_crossentropy',
        'suppobox': 'binary_crossentropy',
        'gozi': 'binary_crossentropy',
        'pykspa': 'binary_crossentropy',
    }

def get_loss_weights():
    return {
        'main': 1.0,
        'matsnu': 0.1,
        'suppobox': 0.1,
        'gozi': 0.1,
        'pykspa': 0.1,
    }

def y_list_to_dict(all_Ys):
    return dict([(label, np.array(y)) for label, y in zip(get_labels(), all_Ys)])
