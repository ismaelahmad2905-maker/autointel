import re
from django import forms
from django.contrib.auth.models import User


class CarDetailsForm(forms.Form):
    make = forms.CharField(widget=forms.HiddenInput())
    model = forms.CharField(widget=forms.HiddenInput())
    year = forms.CharField(widget=forms.HiddenInput())
    problem_text = forms.CharField(
        widget=forms.Textarea(attrs={
            "rows": 4,
            "placeholder": "e.g. battery keeps dying overnight"
        }),
        label="Describe the Issue"
    )

    def clean_problem_text(self):
        problem_text = self.cleaned_data.get("problem_text", "").strip()

        if len(problem_text) < 8:
            raise forms.ValidationError("Please describe the problem in a bit more detail.")

        words = problem_text.split()
        if len(words) < 2:
            raise forms.ValidationError("Please enter a more meaningful symptom description.")

        return problem_text


class RegisterForm(forms.Form):
    email = forms.EmailField(label="Email Address")
    password = forms.CharField(widget=forms.PasswordInput, label="Password")
    confirm_password = forms.CharField(widget=forms.PasswordInput, label="Confirm Password")

    def clean_email(self):
        email = self.cleaned_data.get("email", "").strip().lower()
        if User.objects.filter(username=email).exists():
            raise forms.ValidationError("An account with this email already exists.")
        return email

    def clean_password(self):
        password = self.cleaned_data.get("password")

        if len(password) < 8:
            raise forms.ValidationError("Password must be at least 8 characters.")
        if not re.search(r"\d", password):
            raise forms.ValidationError("Password must contain at least one number.")
        if not re.search(r"[A-Z]", password):
            raise forms.ValidationError("Password must contain at least one capital letter.")

        return password

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        confirm_password = cleaned_data.get("confirm_password")

        if password and confirm_password and password != confirm_password:
            raise forms.ValidationError("Passwords do not match.")

        return cleaned_data


class LoginForm(forms.Form):
    email = forms.EmailField(label="Email Address")
    password = forms.CharField(widget=forms.PasswordInput, label="Password")


class MechanicRegisterForm(forms.Form):
    email = forms.EmailField(label="Email Address")
    code = forms.CharField(label="Mechanic Code")
    password = forms.CharField(widget=forms.PasswordInput, label="Password")
    confirm_password = forms.CharField(widget=forms.PasswordInput, label="Confirm Password")

    def clean_email(self):
        email = self.cleaned_data.get("email", "").strip().lower()
        if User.objects.filter(username=email).exists():
            raise forms.ValidationError("An account with this email already exists.")
        return email

    def clean_password(self):
        password = self.cleaned_data.get("password")

        if len(password) < 8:
            raise forms.ValidationError("Password must be at least 8 characters.")
        if not re.search(r"\d", password):
            raise forms.ValidationError("Password must contain at least one number.")
        if not re.search(r"[A-Z]", password):
            raise forms.ValidationError("Password must contain at least one capital letter.")

        return password

    def clean_code(self):
        code = self.cleaned_data.get("code", "").strip()
        if len(code) < 3:
            raise forms.ValidationError("Please enter a valid mechanic code.")
        return code

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        confirm_password = cleaned_data.get("confirm_password")

        if password and confirm_password and password != confirm_password:
            raise forms.ValidationError("Passwords do not match.")

        return cleaned_data


class MechanicLoginForm(forms.Form):
    email = forms.EmailField(label="Email Address")
    code = forms.CharField(label="Mechanic Code")
    password = forms.CharField(widget=forms.PasswordInput, label="Password")