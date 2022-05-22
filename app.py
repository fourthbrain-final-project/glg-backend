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
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from  ner import bert_ner as bn
from pydantic import BaseModel

class Document(BaseModel):
    document: str

class Topics(BaseModel):
    document: str
    topics: list

app = FastAPI()

@app.post("/classify")
async def classify_document(document: Document):
    classes = bc.classify_document(document.document)
    return { 'label': bc.get_class(classes) }

@app.post("/topics")
async def topic_generator(topics: Topics):
    classes =  bc.classify_document(topics.document, topics.topics)
    return bc.get_top_classes(classes)

@app.post("/entities")
async def get_entites(document: Document):
    entity_dict = bn.get_entities(document.document)
    
    return bn.gather_entities(entity_dict=entity_dict)
