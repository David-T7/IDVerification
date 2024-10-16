from rest_framework import serializers

class IDUploadSerializer(serializers.Serializer):
    front_id_image = serializers.ImageField()
    back_id_image = serializers.ImageField()

class PassportUploadSerializer(serializers.Serializer):
    passport_image = serializers.ImageField()
