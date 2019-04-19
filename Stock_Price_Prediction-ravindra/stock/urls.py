from django.conf.urls import url
from stock import views
from django.urls import path

urlpatterns=[
    path('home/',views.index,name='index'),
    path('chart/',views.index,name='chart'),
    path('output/',views.findList,name='list'),
    path('register/',views.register,name='register'),
    path('user_login/',views.user_login,name='user_login'),
]
#path('home/',views.index,name='index'),
#Model.as_view()