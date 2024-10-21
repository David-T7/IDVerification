from django.db import models
import uuid

class UserImage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)  # Unique identifier for each image
    freelancer_id = models.UUIDField()  # UUID of the freelancer
    user_image = models.ImageField(upload_to='user_images/' , null=True)  # Field to store the image file
    id_image = models.ImageField(upload_to='id_images/' , null=True)  # Field to store the image file
    id_type = models.CharField(max_length=20 , null=True)
    created_at = models.DateTimeField(auto_now_add=True)  # Timestamp for when the image was uploaded

    def __str__(self):
        return f"Image {self.id} for Freelancer {self.freelancer_uuid}"
