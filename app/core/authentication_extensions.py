# authentication_extensions.py
from drf_spectacular.openapi import AbstractOpenApiAuthenticationExtension

class CustomJWTAuthenticationExtension(AbstractOpenApiAuthenticationExtension):
    target_class = 'test.authentication.CustomJWTAuthentication'
    name = 'BearerAuth'

    def get_schema_fields(self, auto_schema):
        return [
            {
                'type': 'http',
                'scheme': 'bearer',
                'bearerFormat': 'JWT',
                'description': 'Enter your JWT token here',
            }
        ]
