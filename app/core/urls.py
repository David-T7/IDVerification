# app/verification/urls.py

from django.urls import path
from . import views
urlpatterns = [
    path('warmup/', views.WarmupView.as_view(), name='warmup'),
    path('verify-id/', views.VerifyIDView.as_view(), name='verify-id'),
    path('verify-passport/', views.VerifyPassportView.as_view(), name='verify-passport'),
    path('verify/face-match/', views.FaceMatchingView.as_view(), name='face-match'),
    path('verify/smile/', views.SmileDetectionView.as_view(), name='smile-detection'),
    path('verify/head-rotation-right/', views.HeadRotationRightView.as_view(), name='head-rotation-right'),
    path('verify/head-rotation-left/', views.HeadRotationLeftView.as_view(), name='head-rotation-left'),
    path('verify/update-user-image/', views.UpdateUserImageView.as_view(), name='update-user-image'),
]
