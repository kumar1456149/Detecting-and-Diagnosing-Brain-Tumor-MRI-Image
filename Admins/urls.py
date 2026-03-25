from django.urls import path
from . import views
urlpatterns = [
    path('', views.AdminLogin , name='adminlogin') , 
    path('adminhome/' , views.AdminHome , name='AdminHome'),
    path('users_view/' , views.User_View , name='User_View'),
    path('activate_user/<int:id>' , views.ActivateUser , name='ActivateUser'),
    path('Delete_User/<int:id>' , views.DeleteUser , name='DeleteUser'),
    path('edit_user/<int:id>' , views.Edit_User , name='Edit_User'),
    path('scan_history/', views.AdminScanHistory, name='AdminScanHistory'),
    path('analytics/', views.AdminAnalytics, name='AdminAnalytics'),
    path('all_reports/', views.AdminReports, name='AdminReports'),
]