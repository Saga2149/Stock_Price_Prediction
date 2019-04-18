from django.db import models

# Create your models here.
class Timestamp(models.Model):
    timeStamp = models.DateField()

    def __str__(self):
        return self.timeStamp.strftime("%Y-%m-%d %H:%M:%S")