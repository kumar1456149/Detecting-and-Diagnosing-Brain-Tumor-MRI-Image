from django.db import models

# Create your models here.

class User_SigUp(models.Model):
    Name = models.CharField(max_length=100)
    Address = models.CharField(max_length=100)
    Mobile = models.CharField(max_length=100)
    Email = models.EmailField(max_length=100)
    Username = models.CharField(max_length=100)
    Password = models.CharField(max_length=100)
    Status = models.CharField(max_length=100)

    def __str__(self):
        return self.Name

class ScanRecord(models.Model):
    user = models.ForeignKey(User_SigUp, on_delete=models.CASCADE, null=True, blank=True)
    image = models.ImageField(upload_to='scans/')
    result = models.CharField(max_length=100)
    accuracy = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.result} - {self.timestamp}"
