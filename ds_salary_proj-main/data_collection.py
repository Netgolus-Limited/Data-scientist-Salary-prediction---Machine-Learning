# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 18:20:22 2022

@author: 97150
"""

import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/PlayingNumbers/ds_salary_proj/master/glassdoor_jobs.csv")
df.to_csv('glassdoor_jobs.csv', index = False)
df.head()

print(df.columns)
