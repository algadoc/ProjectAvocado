# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


# Loading model to compare the results
model = pickle.load(open('RFModel.p','rb'))
print(model.predict([[2, 9, 6]]))