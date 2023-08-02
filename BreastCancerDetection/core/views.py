from django.shortcuts import render
from .models import BreastCancerData,Doctor
# Create your views here.
def index(request):
    datas = BreastCancerData.objects.all()
    doctor_datas = Doctor.objects.all()
    context = {
        'doctors_data':doctor_datas
    }
    return render(request,'index.html',context)
