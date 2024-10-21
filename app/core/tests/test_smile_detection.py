from rest_framework.test import APIClient
from django.urls import reverse
from django.test import TestCase
from unittest.mock import patch
import jwt
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework import status
import os

class SmileDetectionTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.freelancer_id = '0c91fd82-97b1-42ed-a1ef-6fd7e60e3f57'
        # JWT token setup
        self.token = jwt.encode({'freelancer_id': self.freelancer_id}, settings.SECRET_KEY, algorithm='HS256')
        self.url = reverse('smile-detection')  # Ensure this matches your URL configuration

        # Path to test images (adjust the paths accordingly)
        self.smiling_user_image_path = os.path.join(settings.BASE_DIR, 'media', 'images/pics', 'smiling.jpg')
        self.normal_user_image_path = os.path.join(settings.BASE_DIR, 'media', 'images/pics', 'different_user.jpg')

        # Ensure test images exist
        if not os.path.exists(self.smiling_user_image_path):
            raise FileNotFoundError(f"User image not found at {self.smiling_user_image_path}")
        if not os.path.exists(self.normal_user_image_path):
            raise FileNotFoundError(f"User image not found at {self.normal_user_image_path}")

        # Prepare the user image for upload
        with open(self.smiling_user_image_path, 'rb') as img_file:
            self.smiling_user_image = SimpleUploadedFile(
                name='smiling.jpg',
                content=img_file.read(),
                content_type='image/jpeg'
            )
        
        with open(self.normal_user_image_path, 'rb') as img_file:
            self.normal_user_image = SimpleUploadedFile(
                name='different_user.jpg',
                content=img_file.read(),
                content_type='image/jpeg'
            )

    def authenticate(self):
        self.client.credentials(HTTP_AUTHORIZATION='Bearer ' + self.token)

    def test_smile_detection_success(self):
        """
        Test successful smile detection with a valid 'happy' emotion.
        """
        self.authenticate()
        response = self.client.post(
            self.url,
            {'user_image': self.smiling_user_image},
            format='multipart'
        )

        # Assertions
        print("response for success test is ",response.data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertEqual(response.data['status'], 'success')
        self.assertIn('message', response.data)
        self.assertEqual(response.data['message'], 'Smile detected successfully.')

    def test_smile_detection_failure(self):
        """
        Test failed smile detection when the emotion is not 'happy'.
        """
        self.authenticate()
        response = self.client.post(
            self.url,
            {'user_image': self.normal_user_image},
            format='multipart'
        )

        # Assertions
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertEqual(response.data['status'], 'failed')
        self.assertIn('message', response.data)
        self.assertIn("Required emotion 'happy' not detected", response.data['message'])

    def test_smile_detection_no_image(self):
        """
        Test error handling when no user image is provided.
        """
        self.authenticate()

        response = self.client.post(self.url, {}, format='multipart')

        # Assertions
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(str(response.data['user_image'][0]), 'No file was submitted.')

    def test_smile_detection_invalid_data(self):
        """
        Test error handling when invalid data is provided.
        """
        self.authenticate()

        # Posting invalid data (no file)
        response = self.client.post(self.url, {'user_image': 'not_a_file'}, format='multipart')

        # Assertions
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('user_image', response.data)


