from django.contrib import admin
admin.site.site_header = "Shashank Thapa Detection System"
# Register your models here.
from .models import BreastCancerData,Doctor,Patient,PatientDoctor,BreastScanTestResult,PatientBreastScan

admin.site.register([BreastCancerData,Doctor,Patient,PatientDoctor,BreastScanTestResult,PatientBreastScan])