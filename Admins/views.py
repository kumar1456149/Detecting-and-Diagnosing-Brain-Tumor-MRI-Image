from django.shortcuts import render , redirect
from django.contrib import messages
from User.models import User_SigUp, ScanRecord
from django.db.models import Count


# Create your views here.
def AdminLogin(request):
    if request.method=='POST':
        name= request.POST.get('name')
        password = request.POST.get('password')
        print(name and password)
        if name=='admin' and password=='admin':
            return redirect('AdminHome')
        else:
            messages.error(request , 'Invalid Credentails! --Login with Proper Details--')
        

    return render(request , 'AdminLogin.html')

def AdminHome(request):
    users = User_SigUp.objects.all()
    # Calculate global stats
    total_users = users.count()
    total_scans = ScanRecord.objects.count()
    tumor_detected = ScanRecord.objects.filter(result__icontains='Tumor Detected').count()
    no_tumor = ScanRecord.objects.filter(result__icontains='No Tumor').count()
    invalid_scans = total_scans - (tumor_detected + no_tumor)

    context = {
        'users': users,
        'total_users': total_users,
        'total_scans': total_scans,
        'tumor_detected': tumor_detected,
        'no_tumor': no_tumor,
        'invalid_scans': invalid_scans
    }
    return render(request, 'Admins/AdminHome.html', context)

def AdminScanHistory(request):
    scans = ScanRecord.objects.all().order_by('-timestamp')
    return render(request, 'Admins/ScanHistory.html', {'scans': scans})

def AdminAnalytics(request):
    total_scans = ScanRecord.objects.count()
    tumor_count = ScanRecord.objects.filter(result__icontains='Tumor Detected').count()
    healthy_count = ScanRecord.objects.filter(result__icontains='No Tumor').count()
    
    # Calculate distribution
    if total_scans > 0:
        tumor_pct = (tumor_count / total_scans) * 100
        healthy_pct = (healthy_count / total_scans) * 100
    else:
        tumor_pct = healthy_pct = 0

    return render(request, 'Admins/Analytics.html', {
        'tumor_pct': round(tumor_pct, 1),
        'healthy_pct': round(healthy_pct, 1),
        'total_scans': total_scans
    })

def AdminReports(request):
    scans = ScanRecord.objects.all().order_by('-timestamp')
    return render(request, 'Admins/Reports.html', {'scans': scans})


def Edit_User(request, id):
    user = User_SigUp.objects.get(id=id)
    if request.method == 'POST':
        user.Name = request.POST.get('name')
        user.Address = request.POST.get('address')
        user.Mobile = request.POST.get('mobile')
        user.Email = request.POST.get('email')
        user.Username = request.POST.get('username')
        user.Password = request.POST.get('password')
        user.save()
        messages.success(request, 'User updated successfully!')
        return redirect('AdminHome')
    return render(request, 'Admins/Edit_User.html', {'user': user})


def User_View(request):
    users = User_SigUp.objects.all()
    return render(request , 'Admins/User_View.html' , {'users':users})


def ActivateUser(request , id):
    user = User_SigUp.objects.get(id=id)
    if user.Status=='waiting':
     user.Status='active'
     user.save()
    return redirect('AdminHome')

def DeleteUser(request , id):
    user = User_SigUp.objects.get(id=id)
    if user:
        user.delete()
    return redirect('AdminHome')
  


    
       