import os
import csv
import re
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import haversine_distances
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from stylometry.extract import *

def scan_all_dir(path, finalFile):
  with os.scandir(path) as subdirectories:
    for subdirectory in subdirectories:
      data = ""
      unknown_data = ''
      if subdirectory.is_dir():
        with os.scandir(path + subdirectory.name) as fl:
          for file in fl:
            # si existe el merged, borralo
            if file.is_file() and file.name == 'merged.txt':
              print(path + subdirectory.name + '/' + file.name + ' - deleted')
              os.remove(path + subdirectory.name + '/' + file.name)

            elif file.is_file() and file.name != 'unknown.txt':
              with open( path + subdirectory.name + '/' + file.name, encoding='utf-8' ) as fp:
                data += fp.read() + '\n'

            # colecta unknown data 
            if file.is_file() and file.name == 'unknown.txt':
              with open(path + subdirectory.name + '/' + file.name, encoding='utf-8') as fp:
                unknown_data += fp.read() + '\n'
                

        # si no existe archivo merged.txt, procede a crearlo
        if not os.path.exists(path + subdirectory.name + '/' + finalFile):
          with open (path + subdirectory.name + '/' + finalFile, 'w', encoding='utf-8') as fp:
            fp.write(data.lower())
        
        # sobreescribe unknowns con la data en minusculas
        with open (path + subdirectory.name + '/' + 'unknown.txt', 'w', encoding='utf-8') as fp:
            fp.write(unknown_data.lower())



def merge_csv (output_path) :
  merged = pd.read_csv("C:/Users/crisc/OneDrive/Desktop/pruebas_tesis/stylometry-master3/salidafinal/merged_final.csv", encoding='latin1')
  unknown = pd.read_csv("C:/Users/crisc/OneDrive/Desktop/pruebas_tesis/stylometry-master3/salidafinal/unknown.csv", encoding='latin1').drop('Author', axis=1)

  merged_csv = pd.concat([merged, unknown], axis=1)
  merged_csv.to_csv(output_path, index=False,encoding='latin1')          


def all_authors (path, n_words):
  novel_corpus = StyloCorpus.from_glob_pattern(path+'*/merged.txt', n_words)
  novel_corpus.output_csv('C:/Users/crisc/OneDrive/Desktop/pruebas_tesis/stylometry-master3/salidafinal/merged_final.csv')
  unknown_corpus = StyloCorpus.from_glob_pattern(path+'*/unknown.txt', n_words)
  unknown_corpus.output_csv('C:/Users/crisc/OneDrive/Desktop/pruebas_tesis/stylometry-master3/salidafinal/unknown.csv')


def compose_array(line, half):
  arr = line[0].split(',')
  real_arr = []
  
  for num in arr:
    try:        
      real_arr.append(float(num))
    except:
      pass

  length = len(real_arr) 
  
  middle_index = length//2

  if half == 1:
    return real_arr[:middle_index]
  elif half == 2:
    return real_arr[middle_index:]
  else :
    return null


def add_methods(pairwise_distances_methods, cdist_methods, output_path):  

  df = pd.read_csv(output_path, encoding='latin1')  
  
  # adding headers to csv
  for idx, distance in enumerate(pairwise_distances_methods + cdist_methods):
    df[distance] = ''
  df.to_csv(output_path, index=False, encoding='latin1')

  with open(output_path, "r") as f:
    reader = csv.reader(f, delimiter="\t")
        
    for i, line in enumerate(reader):      

      # primera y segunda mitad del arreglo
      first_half = compose_array(line, 1)
      second_half = compose_array(line, 2)

      final_array = []

      
      if len(first_half) and len(second_half):

        
        dataX = np.array([first_half])
        dataY = np.array([second_half]) 

        dataXV = dataX.reshape(-1, 1)
        dataYV = dataY.reshape(-1, 1)
        
        # normalization start
        scalerX = MinMaxScaler().fit_transform(dataXV)
        scalerY = MinMaxScaler().fit_transform(dataYV)

        X = scalerX.reshape(1, -1)
        Y = scalerY.reshape(1, -1)
        # normalization end

        for distance in pairwise_distances_methods: 
          if distance == 'chi2': 
            value = chi2_kernel(X, Y)[0][0]
            df.at[i-1, distance] = value
            final_array.append(value)                   
          else :
            value = pairwise_distances(X,Y, metric=distance)[0][0]
            df.at[i-1, distance] = value
            final_array.append(value)

        
        for metric in cdist_methods:               
          if metric == 'final_array':
            df.at[i-1, metric] = final_array
          else :
            value = cdist(X, Y, metric=metric)[0][0]
            df.at[i-1, metric] = value
            final_array.append(value)

  df.to_csv(output_path, index=False, encoding='latin1')


def truth_file (output_path, truth_path):

  #truth file
  df = pd.read_csv(output_path, encoding='latin1')
  df['truth'] = ''
  df['truth_binary'] = ''
  
  with open(output_path, "r") as f:
    reader = csv.reader(f, delimiter="\t")

    with open(truth_path) as f:
      lines = [line.rstrip('\n') for line in f]
      
      for i, line in enumerate(lines):
        df.at[i, 'truth'] = line.split(' ')[1]
        df.at[i, 'truth_binary'] = 1 if line.split(' ')[1] == 'Y' else 0

  df.to_csv(output_path, index=False, encoding='latin1')