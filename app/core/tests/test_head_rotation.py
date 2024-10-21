from rest_framework.test import APIClient
from django.urls import reverse
from django.test import TestCase
import jwt
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework import status
import os

class HeadRotationTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.freelancer_id = '0c91fd82-97b1-42ed-a1ef-6fd7e60e3f57'
        # JWT token setup
        self.token = jwt.encode({'freelancer_id': self.freelancer_id}, settings.SECRET_KEY, algorithm='HS256')
        self.right_url = reverse('head-rotation-right')  # Ensure this matches your URL configuration
        self.left_url = reverse('head-rotation-left')  # Ensure this matches your URL configuration
        # Create test images for head rotation detection
        self.right_user_image_path = os.path.join(settings.BASE_DIR, 'media', 'images/pics', 'right2.jpg')
        self.left_user_image_path = os.path.join(settings.BASE_DIR, 'media', 'images/pics', 'left2.jpg')

        # Ensure test images exist
        if not os.path.exists(self.right_user_image_path):
            raise FileNotFoundError(f"Right User image not found at {self.right_user_image_path}")
        if not os.path.exists(self.left_user_image_path):
            raise FileNotFoundError(f"Left User image not found at {self.left_user_image_path}")

        # Prepare the user image for upload
        with open(self.right_user_image_path, 'rb') as img_file:
            self.right_user_image = SimpleUploadedFile(
                name='right2.jpg',
                content=img_file.read(),
                content_type='image/jpeg'
            )
        # Prepare the user image for upload
        with open(self.left_user_image_path, 'rb') as img_file:
            self.left_user_image = SimpleUploadedFile(
                name='left2.jpg',
                content=img_file.read(),
                content_type='image/jpeg'
            )

    def authenticate(self):
        self.client.credentials(HTTP_AUTHORIZATION='Bearer ' + self.token)

    def test_head_rotation_right_success(self):
        """
        Test successful head rotation detection.
        """
        self.authenticate()
        response = self.client.post(
            self.right_url,
            {
                'user_image': self.right_user_image,
                'yaw_threshold': 20 , # Example threshold
                'freelancer_id':self.freelancer_id

            },
            format='multipart'
        )

        # Assertions
        print("rotation sucess data",response.data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertEqual(response.data['status'], 'success')
        self.assertIn('message', response.data)
        self.assertEqual(response.data['message'], 'Head rotation detected successfully.')
    def test_head_rotation_left_success(self):
        """
        Test successful head rotation detection.
        """
        self.authenticate()
        response = self.client.post(
            self.left_url,
            {
                'user_image': self.left_user_image,
                'yaw_threshold': 20,  # Example threshold
                'freelancer_id':self.freelancer_id

            },
            format='multipart'
        )

        # Assertions
        print("rotation sucess data",response.data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertEqual(response.data['status'], 'success')
        self.assertIn('message', response.data)
        self.assertEqual(response.data['message'], 'Head rotation detected successfully.')

    def test_head_rotation_missing_image(self):
        """
        Test error handling when no user image is provided.
        """
        self.authenticate()
        response = self.client.post(self.right_url, {}, format='multipart')
        print("rotation missing image data",response.data)

        # Assertions
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(str(response.data['user_image'][0]), 'No file was submitted.')

    def test_head_rotation_right_failed_detection(self):
        """
        Test scenario where head rotation requirement is not met.
        """
        self.authenticate()
        # Assuming we have a way to simulate failed detection
        # You might want to modify the user_image or create a mock to simulate this
        response = self.client.post(
            self.right_url,
            {
                'user_image': self.left_user_image,
                'yaw_threshold': 20,  # Use a threshold to determine rotation
                'freelancer_id':self.freelancer_id

            },
            format='multipart'
        )
        print("rotation failed to detect data",response.data)


        # Assertions
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertEqual(response.data['status'], 'failed')
        self.assertIn('message', response.data)
        self.assertEqual(response.data['message'], "Head rotation requirement not met. Please rotate your head to the right.")
    
    def test_head_rotation_left_failed_detection(self):
        """
        Test scenario where head rotation requirement is not met.
        """
        self.authenticate()
        # Assuming we have a way to simulate failed detection
        # You might want to modify the user_image or create a mock to simulate this
        response = self.client.post(
            self.left_url,
            {
                'user_image': self.right_user_image,
                'yaw_threshold': 20,  # Use a threshold to determine rotation
                'freelancer_id':self.freelancer_id

            },
            format='multipart'
        )
        print("rotation failed to detect data",response.data)


        # Assertions
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertEqual(response.data['status'], 'failed')
        self.assertIn('message', response.data)
        self.assertEqual(response.data['message'], "Head rotation requirement not met. Please rotate your head to the left.")


