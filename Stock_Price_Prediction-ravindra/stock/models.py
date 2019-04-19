from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Timestamp(models.Model):
    timeStamp = models.DateField()

    def __str__(self):
        return self.timeStamp.strftime("%Y-%m-%d %H:%M:%S")


class UserProfileInfo(models.Model):

    user = models.OneToOneField(User, on_delete=models.CASCADE)

    portfolio_site = models.URLField(blank=True)

    def __str__(self):
        return self.user.username
