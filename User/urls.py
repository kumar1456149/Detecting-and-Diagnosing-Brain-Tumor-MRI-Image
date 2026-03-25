from django.urls import path
from . import views

urlpatterns = [
    path('', views.SigUp , name='userlogin') ,
    path('UserLogin/' , views.UserLogin , name='UserLogin'),
    path('User_Home/' , views.UserHome , name='UserHome'),
    path('Traning/' , views.Traning , name='Traning'),
    path('predict/' , views.predict , name='predict'),
    path('performance/' , views.performance , name='performance'),
    path('probability/' , views.probability , name='probability'),
    path('history/' , views.history , name='history'),
    path('reports/' , views.reports , name='reports'),
    path('download_report/<int:id>/', views.download_report, name='download_report'),
]