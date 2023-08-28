# classifier_app/urls.py
from django.urls import path
from . import views
# from .views import X_test
urlpatterns = [
    path('predict_audio', views.predict_audio, name='predict_audio'),
    path('home', views.home, name='home'),
    path('progress', views.progress, name='progress'),
    path('confusion', views.confusion_matrix_view, name='confusion'),
]
