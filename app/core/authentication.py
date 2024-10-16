import jwt
from rest_framework.authentication import BaseAuthentication
from rest_framework import exceptions
from django.conf import settings
from rest_framework.permissions import BasePermission


class CustomJWTAuthentication(BaseAuthentication):
    '''
        Custom authentication class for DRF and JWT
    '''

    def authenticate(self, request):
        authorization_header = request.headers.get('Authorization')
        if not authorization_header:
            return None

        try:
            # Split the Authorization header to get the token
            access_token = authorization_header.split(' ')[1]
            print("Access token is:", access_token)

            # Decode the token
            payload = jwt.decode(access_token, settings.SECRET_KEY, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            msg = 'Signature has expired.'
            raise exceptions.AuthenticationFailed(msg)
        except jwt.DecodeError:
            msg = 'Error decoding signature.'
            raise exceptions.AuthenticationFailed(msg)
        except jwt.InvalidTokenError:
            raise exceptions.AuthenticationFailed('Invalid token.')

        # Return the payload if token is valid
        return (None, payload)

    def authenticate_header(self, request):
        """
        Return a string to be used as the value of the `WWW-Authenticate`
        header in a `401 Unauthenticated` response.
        """
        return 'Bearer'



class TokenPayloadPermission(BasePermission):
    """
    Custom permission class that checks the token payload for specific attributes.
    """

    def has_permission(self, request, view):
        # Access token is passed in headers
        auth = request.headers.get('Authorization')
        if not auth:
            return False

        token = auth.split(' ')[1]
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return False
        except jwt.DecodeError:
            return False

        return True
        