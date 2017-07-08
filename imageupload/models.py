from __future__ import unicode_literals

from django.db import models
import uuid
def scrable_uploaded_filename(instance,filename):
    extension= filename.split(".")[-1]
    return "{}.{}".format(uuid.uuid4(),extension)

class UploadImage(models.Model):
    image = models.ImageField('uploaded_image',upload_to=scrable_uploaded_filename) # Stores the uploaded image
