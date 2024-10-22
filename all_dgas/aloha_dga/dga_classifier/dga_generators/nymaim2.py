#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 02:03:12 2020

@author: ighosh
"""

import json
import argparse
from datetime import datetime
import hashlib


class Rand:

    def __init__(self, seed, year, yday, offset=0):
        m = self.md5(seed)
        s = "{}{}{}".format(m, year, yday + offset)
        self.hashstring = self.md5(s)

    @staticmethod
    def md5(s):
        return hashlib.md5(s.encode('ascii')).hexdigest()

    def getval(self):
        v = int(self.hashstring[:8], 16)
        self.hashstring = self.md5(self.hashstring[7:])
        return v

def dga(date,include_tld=True):
    with open("words.json", "r") as r:
        wt = json.load(r)

    seed = "3138C81ED54AD5F8E905555A6623C9C9"
    daydelta = 10
    maxdomainsfortry = 64
    year = date.year % 100
    yday = date.timetuple().tm_yday - 1
    result=[]
    for dayoffset in range(daydelta + 1):
        r = Rand(seed, year, yday - dayoffset)
        for _ in range(maxdomainsfortry):
            domain = ""
            if include_tld:
                temp = ['firstword', 'separator', 'secondword', 'tld']
            else:
                temp = ['firstword', 'separator', 'secondword']
            for s in temp:
                ss = wt[s]
                domain += ss[r.getval() % len(ss)]
            print(domain)
            result.append(domain)
    return result
#include a tld case so as to remove tld if necessary            
            
def generate_dga(num_domains,include_tld=True):
        date = datetime.now()
        ret=[x for x in dga(date,include_tld)]
        return ret[:min(num_domains,len(ret))]        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--date", help="as YYYY-mm-dd")
    args = parser.parse_args()
    date_str = args.date
    if date_str:
        date = datetime.strptime(date_str, "%Y-%m-%d")
    else:
        date = datetime.now() 
    dga(date) 
#Debugging and linking left