from django.conf.urls import url, include
from stock import views
from django.urls import path

urlpatterns=[
    path('home/',views.index,name='index'),
    path('chart/',views.index,name='chart'),
    path('output/',views.findList,name='list'),
    path('register/',views.register,name='register'),
    path('login/',views.login,name='login'),
]
#path('home/',views.index,name='index'),
#Model.as_view()