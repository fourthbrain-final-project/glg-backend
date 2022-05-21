#!/usr/bin/python

'''
    Created by: Brett and Jose
    Updated by: Brett and Jose
    Created on: 2022-05-21
    Updated on: 2022-05-21
    Purpose: Driver for REST API that processes calls and passes them to middle tier
'''

from classifier import bart_classifier as bc
from fastapi import FastAPI
from  ner import bert_ner as bn
from pydantic import BaseModel

class BartClassifier(BaseModel):
    class_label: str

class Topic(BaseModel):
    class_label: str
    class_probability: float

class Topics(BaseModel):
    topics: list(Topic)

class Document(BaseModel):
    document: str

app = FastAPI()

