# app/verification/urls.py

from django.urls import path
from .views import VerifyIDView , VerifyPassportView , FaceMatchingView , SmileDetectionView , HeadRotationLeftView , HeadRotationRightView

urlpatterns = [
    path('verify-id/', VerifyIDView.as_view(), name='verify-id'),
    path('verify-passport/', VerifyPassportView.as_view(), name='verify-passport'),
    path('verify/face-match/', FaceMatchingView.as_view(), name='face-match'),
    path('verify/smile/', SmileDetectionView.as_view(), name='smile-detection'),
    path('verify/head-rotation-right/', HeadRotationRightView.as_view(), name='head-rotation-right'),
    path('verify/head-rotation-left/', HeadRotationLeftView.as_view(), name='head-rotation-left'),
]
