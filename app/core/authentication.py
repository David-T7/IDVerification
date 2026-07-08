import logging

import jwt
from rest_framework.authentication import BaseAuthentication
from rest_framework import exceptions
from django.conf import settings
from rest_framework.permissions import BasePermission

logger = logging.getLogger(__name__)


class CustomJWTAuthentication(BaseAuthentication):
    '''Custom authentication class for DRF and JWT'''

    def authenticate(self, request):
        authorization_header = request.headers.get('Authorization')
        if not authorization_header:
            return None

        parts = authorization_header.split(' ')
        if len(parts) != 2:
            raise exceptions.AuthenticationFailed('Invalid Authorization header format.')

        try:
            payload = jwt.decode(parts[1], settings.SECRET_KEY, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            raise exceptions.AuthenticationFailed('Signature has expired.')
        except jwt.DecodeError:
            raise exceptions.AuthenticationFailed('Error decoding signature.')
        except jwt.InvalidTokenError:
            raise exceptions.AuthenticationFailed('Invalid token.')

        return (None, payload)

    def authenticate_header(self, request):
        return 'Bearer'


class TokenPayloadPermission(BasePermission):
    """Custom permission class that checks the token payload."""

    def has_permission(self, request, view):
        auth = request.headers.get('Authorization')
        if not auth:
            return False

        parts = auth.split(' ')
        if len(parts) != 2:
            return False

        try:
            jwt.decode(parts[1], settings.SECRET_KEY, algorithms=['HS256'])
        except (jwt.ExpiredSignatureError, jwt.DecodeError, jwt.InvalidTokenError):
            return False

        return True
