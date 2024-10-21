from rest_framework.test import APIClient
from django.urls import reverse
from django.test import TestCase
import jwt
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework import status
import os

class FaceMatchingTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.freelancer_id = '0c91fd82-97b1-42ed-a1ef-6fd7e60e3f57'
        # JWT token setup
        self.token = jwt.encode({'freelancer_id': self.freelancer_id}, settings.SECRET_KEY, algorithm='HS256')
        self.url = reverse('face-match')  # Ensure this matches your URL configuration

        # Create test images for face matching
        self.id_image_path = os.path.join(settings.BASE_DIR, 'media', 'images/test', 'front_id.jpg')
        self.user_image_path = os.path.join(settings.BASE_DIR, 'media', 'images/pics', 'front.jpg')
        self.different_image_path = os.path.join(settings.BASE_DIR, 'media', 'images/pics', 'different_user.jpg')

        # Ensure test images exist
        if not os.path.exists(self.id_image_path):
            raise FileNotFoundError(f"ID image not found at {self.id_image_path}")
        if not os.path.exists(self.user_image_path):
            raise FileNotFoundError(f"User image not found at {self.user_image_path}")
        if not os.path.exists(self.different_image_path):
            raise FileNotFoundError(f"Different user image not found at {self.different_image_path}")

        # Prepare the ID image for upload
        with open(self.id_image_path, 'rb') as img_file:
            self.id_image = SimpleUploadedFile(
                name='front_id.jpg',
                content=img_file.read(),
                content_type='image/jpeg'
            )

        # Prepare the user image for upload
        with open(self.user_image_path, 'rb') as img_file:
            self.user_image = SimpleUploadedFile(
                name='front.jpg',
                content=img_file.read(),
                content_type='image/jpeg'
            )
        
        # Prepare the different user image for upload
        with open(self.different_image_path, 'rb') as img_file:
            self.different_user_image = SimpleUploadedFile(
                name='different_user.jpg',
                content=img_file.read(),
                content_type='image/jpeg'
            )

    def authenticate(self):
        self.client.credentials(HTTP_AUTHORIZATION='Bearer ' + self.token)

    def test_face_match_success(self):
        """
        Test successful face matching with valid images.
        """
        self.authenticate()
        response = self.client.post(
            self.url,
            {
                'id_image': self.id_image,
                'user_image': self.user_image,
                'freelancer_id':self.freelancer_id
            },
            format='multipart'
        )

        # Assertions
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertEqual(response.data['status'], 'success')
        self.assertIn('message', response.data)
        self.assertEqual(response.data['message'], 'Face match successful.')

    def test_face_mismatch(self):
        """
        Test a face mismatch scenario where the images don't match.
        """
        self.authenticate()
        # Assume we can alter the user image to simulate mismatch (use another valid image here)
        with open(self.different_image_path, 'rb') as img_file:
            different_user_image = SimpleUploadedFile(
                name='different_user.jpg',
                content=img_file.read(),
                content_type='image/jpeg'
            )

        response = self.client.post(
            self.url,
            {
                'id_image': self.id_image,
                'user_image': different_user_image,
                'freelancer_id':self.freelancer_id

            },
            format='multipart'
        )

        # Assertions
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertEqual(response.data['status'], 'failed')
        self.assertIn('message', response.data)
        self.assertIn('Distance:', response.data['message'])  # Ensure distance is returned

    def test_face_match_missing_images(self):
        self.authenticate()
        response = self.client.post(self.url, {}, format='multipart')

        # Assertions
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('user_image', response.data)
        self.assertEqual(str(response.data['user_image'][0]), 'No file was submitted.')


