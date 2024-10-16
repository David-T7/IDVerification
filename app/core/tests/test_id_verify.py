from rest_framework.test import APIClient
from django.urls import reverse
from django.utils import timezone
from django.test import TestCase
import jwt
from django.conf import settings
import os
from rest_framework import status  # Importing status here
from django.core.files.uploadedfile import SimpleUploadedFile
import tempfile
from PIL import Image

class IDVerificationTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.freelancer_id = '0c91fd82-97b1-42ed-a1ef-6fd7e60e3f57'
        # JWT token setup
        self.token = jwt.encode({'freelancer_id': self.freelancer_id}, settings.SECRET_KEY, algorithm='HS256')
        self.url = reverse('verify-id')  # Ensure this name matches your URL configuration
        self.pass_url = reverse('verify-passport')  # Ensure this name matches your URL configuration
         # Create a real image for testing using PIL
        # Paths to the real images
        self.front_id_source_path = os.path.join(settings.BASE_DIR, 'media', 'images/test', 'front_id.jpg')
        self.back_id_source_path = os.path.join(settings.BASE_DIR, 'media', 'images/test', 'back_id.jpg')
        self.passport_image_path = os.path.join(settings.BASE_DIR, 'media', 'images/test', 'passport.jpg')

        # Ensure the images exist
        if not os.path.exists(self.front_id_source_path):
            raise FileNotFoundError(f"Front ID image not found at {self.front_id_source_path}")
        if not os.path.exists(self.back_id_source_path):
            raise FileNotFoundError(f"Back ID image not found at {self.back_id_source_path}")
        if not os.path.exists(self.passport_image_path):
            raise FileNotFoundError(f"Passport image not found at {self.passport_image_path}")
        
        # Prepare the front ID image for upload
        with open(self.front_id_source_path, 'rb') as img_file:
            self.front_id_image = SimpleUploadedFile(
                name='front_id.jpg',
                content=img_file.read(),
                content_type='image/jpeg'
            )
        
        # Prepare the back ID image for upload
        with open(self.back_id_source_path, 'rb') as img_file:
            self.back_id_image = SimpleUploadedFile(
                name='back_id.jpg',
                content=img_file.read(),
                content_type='image/jpeg'
            )
          # Prepare the back ID image for upload
        with open(self.passport_image_path, 'rb') as img_file:
            self.passport_image = SimpleUploadedFile(
                name='passport.jpg',
                content=img_file.read(),
                content_type='image/jpeg'
            )
    def authenticate(self):
        self.client.credentials(HTTP_AUTHORIZATION='Bearer ' + self.token)

    def test_verify_id_success(self):
        """
        Test the successful verification of an ID with all steps passing.
        """
        self.authenticate()
        # Make the POST request with the valid image
        response = self.client.post(
            self.url,
            {
                'front_id_image': self.front_id_image,
                'back_id_image': self.back_id_image
            },
            format='multipart'
        )

        # Assertions
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('authenticity_score', response.data)
        self.assertIn('details', response.data)
        
        # Check the authenticity score is within expected range
        self.assertTrue(0 <= response.data['authenticity_score'] <= 100)
        
        # Check details
        details = response.data['details']
        self.assertIn('ELA_Tampered', details)
        self.assertIn('ELA_Score', details)
        self.assertIn('Clone_Detected', details)
        self.assertIn('Clone_Count', details)
        self.assertIn('Forgery_Detected', details)
        self.assertIn('Forgery_Confidence', details)
        self.assertIn('Anomaly_Detected', details)
        self.assertIn('MSE', details)
        self.assertIn('Personal_Info_Match', details)
        
        personal_info = details['Personal_Info_Match']
        self.assertIn('Is_Match', personal_info)
        self.assertIn('Match_Percentage', personal_info)
        self.assertIn('Match_Score', personal_info)
        self.assertIn('Total_Fields', personal_info)
        self.assertIn('Mismatches', personal_info)
        
        # Optionally, further assertions based on expected values
        # For example:
        # self.assertGreaterEqual(details['authenticity_score'], 75)
    def test_verify_passport_success(self):
        """
        Test the successful verification of an passport with all steps passing.
        """
        self.authenticate()
        # Make the POST request with the valid image
        response = self.client.post(
            self.pass_url,
            {
                'passport_image': self.passport_image
            },
            format='multipart'
        )

        # Assertions
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('authenticity_score', response.data)
        self.assertIn('details', response.data)
        
        # Check the authenticity score is within expected range
        self.assertTrue(0 <= response.data['authenticity_score'] <= 100)
        
        # Check details
        details = response.data['details']
        self.assertIn('ELA_Tampered', details)
        self.assertIn('ELA_Score', details)
        self.assertIn('Clone_Detected', details)
        self.assertIn('Clone_Count', details)
        self.assertIn('Forgery_Detected', details)
        self.assertIn('Forgery_Confidence', details)
        self.assertIn('Anomaly_Detected', details)
        self.assertIn('MSE', details)  
        # Optionally, further assertions based on expected values
        # For example:
        # self.assertGreaterEqual(details['authenticity_score'], 75)
    
    def test_verify_id_model_loading_failure(self):
        """
        Test handling of model loading failures.
        """
        self.authenticate()
        # Temporarily rename the model files to simulate failure
        original_forgery_model = os.path.join(settings.MODELS_ROOT, 'forgery_detection_model.keras')
        backup_forgery_model = os.path.join(settings.MODELS_ROOT, 'forgery_detection_model_backup.keras')
        original_autoencoder_model = os.path.join(settings.MODELS_ROOT, 'autoencoder_model.keras')
        backup_autoencoder_model = os.path.join(settings.MODELS_ROOT, 'autoencoder_model_backup.keras')
        
        try:
            # Rename the models to simulate missing files
            os.rename(original_forgery_model, backup_forgery_model)
            os.rename(original_autoencoder_model, backup_autoencoder_model)
            
            # Make the POST request with the valid image
            response = self.client.post(
            self.url,
            {
                'front_id_image': self.front_id_image,
                'back_id_image': self.back_id_image
            },
            format='multipart'
            )
            
            # Assertions
            self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
            self.assertIn('error', response.data)
            self.assertIn('Error loading models', response.data['error'])
        
        finally:
            # Restore the original model files
            os.rename(backup_forgery_model, original_forgery_model)
            os.rename(backup_autoencoder_model, original_autoencoder_model)
    # def tearDown(self):
    #     """
    #     Clean up any files created during tests.
    #     """
    #     upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    #     if os.path.exists(upload_dir):
    #         for filename in os.listdir(upload_dir):
    #             file_path = os.path.join(upload_dir, filename)
    #             if os.path.isfile(file_path):
    #                 os.remove(file_path)