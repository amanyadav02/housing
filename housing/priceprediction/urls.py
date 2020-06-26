from django.contrib import admin
from django.urls import path
from priceprediction import views

urlpatterns = [
    path('',views.index,name='home'),
    path('about/',views.about,name='about'),
    path('services/',views.services,name='services'),
    path('contact/',views.contact,name='contact'),
    path('login/',views.logindevta,name='login'),
    path('logout/',views.logoutdevta,name='logout'),
    path('register/',views.register,name='register'),
    path('home/',views.home,name='home'),
]
