# app/verification/urls.py

from django.urls import path
from .views import VerifyIDView , VerifyPassportView

urlpatterns = [
    path('verify-id/', VerifyIDView.as_view(), name='verify-id'),
    path('verify-passport/', VerifyPassportView.as_view(), name='verify-passport'),
]
