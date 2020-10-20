from tkinter import *
from flask import Flask,redirect, url_for,render_template,request
import os

def d_dtcn():
        os.system("python drowsiness_detection.py --shape_predictor shape_predictor_68_face_landmarks.dat")
        exit()
