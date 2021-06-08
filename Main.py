import pandas
import sklearn

from os.path import join
from os import listdir
from sklearn.model_selection import train_test_split
# 1 Parameters
# TODO: set parameters

# 2 Importing dataset
Num_classes = 10

data_dir = 'input/'
labels = pandas.read_csv(join(data_dir, 'labels.csv'))

print("Calkowita ilosc obrazow treningowych: {}".format(len(listdir(join(data_dir, 'train')))))
# Wyswietlanie Num_classes - ilości ras
print("Pierwsze {} ras".format(Num_classes))
print(labels
      .groupby("breed")
      .count()
      .sort_values("id", ascending=False)
      .head(Num_classes)
      )
# TODO: add cross validation
