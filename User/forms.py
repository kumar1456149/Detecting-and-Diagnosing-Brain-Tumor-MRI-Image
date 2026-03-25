from django import forms

from .models import User_SigUp

class User_SigupForm(forms.ModelForm):
    Name = forms.CharField(widget=(forms.TextInput(
                      attrs={'class':'form-control' , 
                       'placeholder':'Name' ,
                       'autocomplete' :'off', 
                       'title':'Only Alphabets' ,
                        'pattern':'[a-zA-Z]*' })),max_length=100)

    Address = forms.CharField(widget=(forms.TextInput(
                attrs={'class':'form-control' , 
                       'placeholder':'Address' ,
                       'autocomplete' :'off',                          
                         }))   ,max_length=100)
    
    Mobile = forms.CharField(widget=(forms.TextInput(
                attrs={'class':'form-control' , 
                       'placeholder':'Mobile' ,
                       'autocomplete' :'off', 
                       'title':'Enter 10 digit mobile number',
                        'pattern':'[0-9]{10}' } ) ),max_length=10)
    
    Email = forms.EmailField(widget=(forms.EmailInput(
                attrs={'class':'form-control' , 
                       'placeholder':'Email' ,
                       'autocomplete' :'off',
                        'title':'Enter Valid Email',
                        'pattern':'[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$' } )),max_length=100)
    
    Username = forms.CharField(widget=(forms.TextInput(
                attrs={'class':'form-control' ,
                        'placeholder':'Username' ,
                        'autocomplete' :'off', 
                        'title':'Only Alphabets' , 
                        'pattern':'[a-zA-Z]*' } ) ),max_length=100)
    
    Password = forms.CharField(widget=(forms.PasswordInput(
                   attrs={'class':'form-control' , 'placeholder':'Password' ,'autocomplete' :'off',
                        'title':'Must contain at least 8 chars, 1 uppercase, 1 lowercase, 1 number, and 1 special char'
                          ,'pattern': '^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&#])[A-Za-z\d@$!%*?&#]{8,}$' }
                          )),max_length=100)
    
    Status = forms.CharField(widget=(forms.HiddenInput(
        attrs={'class':'form-control' ,
                'placeholder':'Status' ,
                'autocomplete' :'off' })),max_length=100 , initial='waiting')

    class Meta:
        model = User_SigUp
        fields = '__all__'

    def clean_Mobile(self):
        mobile = self.cleaned_data.get('Mobile')
        if not mobile.isdigit():
            raise forms.ValidationError("Mobile number must contain only digits")
        if len(mobile) != 10:
            raise forms.ValidationError("Mobile number must be exactly 10 digits")
        return mobile

    def clean_Password(self):
        password = self.cleaned_data.get('Password')
        import re
        if len(password) < 8:
            raise forms.ValidationError("Password must be at least 8 characters long")
        if not re.search("[a-z]", password):
            raise forms.ValidationError("Password must contain at least one lowercase letter")
        if not re.search("[A-Z]", password):
            raise forms.ValidationError("Password must contain at least one uppercase letter")
        if not re.search("[0-9]", password):
            raise forms.ValidationError("Password must contain at least one number")
        if not re.search("[@$!%*?&#]", password):
            raise forms.ValidationError("Password must contain at least one special character")
        return password



