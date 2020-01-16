#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:30:47 2020

@author: ighosh
"""
from datetime import datetime
from io import BytesIO,StringIO
#import urllib
from urllib import urlopen
from zipfile import ZipFile
import requests
import pickle as pickle
import os
import random
import tldextract
import numpy as np

ALEXA_1M = 'http://s3.amazonaws.com/alexa-static/top-1m.csv.zip'

url = urlopen(ALEXA_1M)
zipfile = ZipFile(BytesIO(url.read()))
print([tldextract.extract(x.split(',')[1]).domain for x in ((zipfile.read(filename).decode('utf-8'))).split()[:num]])
