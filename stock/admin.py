from django.contrib import admin
from stock.models import Timestamp
from stock.models import UserProfileInfo

# Register your models here.

admin.site.register(Timestamp)
admin.site.register(UserProfileInfo)
