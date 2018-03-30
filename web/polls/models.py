# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

from django.utils import timezone
import datetime

# Create your models here.

class Question(models.Model):
    def __str__(self):
        return self.question_text

    question_text=models.CharField(max_length=200)
    pup_data=models.DateTimeField('date published')
    
    def was_published_recently(self):
        return self.pup_data.now()-datetime.timedelta(days=1)

class Choice(models.Model):
    def __str__(self):
        return self.choice_text 
    question=models.ForeignKey(Question,on_delete=models.CASCADE)
    choice_text=models.CharField(max_length=200)
    votes=models.IntegerField(default=0)
