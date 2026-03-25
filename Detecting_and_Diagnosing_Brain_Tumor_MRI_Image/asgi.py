"""
ASGI config for Detecting_and_Diagnosing_Brain_Tumor_MRI_Image project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detecting_and_Diagnosing_Brain_Tumor_MRI_Image.settings')

application = get_asgi_application()
