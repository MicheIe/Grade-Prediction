from flask import Flask, request, render_template
import numpy as np
import sys
import os
import pickle
import sklearn

scalerfile = '/Users/apple/Documents/model/scaler.sav'
scaler = pickle.load(open(scalerfile, 'rb'))
scalerfiley ='/Users/apple/Documents/model/scalery.sav'
scalery = pickle.load(open(scalerfiley,'rb'))

sys.path.append(os.path.abspath('./model'))


from load import *
global model, graph
model,graph = init()

app = Flask(__name__)

def prepareGrades(grades):
    x =[5,5,3,2]
    grades = grades.split(',')
    grades = np.asarray(grades)
    grades = grades.astype('float64')
    avg = np.mean(grades)
    x = np.asarray(x)
    grades = np.append(x,grades)
    grades = np.append(grades,avg)
    grade = getGrade2(avg)
    FinalGrade = FINALGRADE2(grade)
    grades = np.append(grades,FinalGrade)
    grades = grades.reshape(1,-1)
    grades = scaler.transform(grades)
    return grades

def getGrade2(avg):
    if avg >= 80:
        Grade = ('A*')
    elif avg >= 70:
        Grade = ('A')
    elif avg >= 60:
        Grade = ('B')
    else: 
        Grade = ('C')
    return Grade
def FINALGRADE2(p):
    switcher={
            'A*':4,
            'A':3,
            'B':2,
            'C':1
            }
    FinalGrade = switcher.get(p)
    return FinalGrade


@app.route('/')
def index():
    return render_template('gradeForm.html')

@app.route('/', methods=['POST'])
def predict():
    grades = request.form['Grades']
    inputGrade = prepareGrades(grades)
    with graph.as_default():
        outEncoded = model.predict(inputGrade)
        outDecoded = np.round(scalery.inverse_transform(outEncoded))
        outDecoded = np.array_str(outDecoded)
        out = outDecoded
        return out
    
if __name__ == '__main__':
    app.debug=True
    app.run(host='0.0.0.0',port=5000)
        


    
    
