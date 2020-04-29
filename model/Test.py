import pickle
scalerfile = 'scaler.sav'
scaler = pickle.load(open(scalerfile, 'rb'))
import numpy as np

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

grades = '100.0,100.0,100.0,98.0,100.0,100.0,97.0,100.0,89.0,97.0,95.0703125,95.0,77.0,93.0,95.0703125,84.0,96.0'
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




