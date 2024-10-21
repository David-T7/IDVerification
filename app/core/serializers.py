from rest_framework import serializers

class IDUploadSerializer(serializers.Serializer):
    front_id_image = serializers.ImageField()
    back_id_image = serializers.ImageField()

class PassportUploadSerializer(serializers.Serializer):
    passport_image = serializers.ImageField()

class BiometricImageUploadSerializer(serializers.Serializer):
    id_image = serializers.ImageField(required=False)  # For face matching
    user_image = serializers.ImageField()  # For face matching, smile detection, or head rotation
    