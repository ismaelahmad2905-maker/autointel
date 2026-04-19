from django.db import models
from django.contrib.auth.models import User


class Part(models.Model):
    make = models.CharField(max_length=100)
    model = models.CharField(max_length=100)
    year = models.IntegerField()
    category = models.CharField(max_length=100)
    part_name = models.CharField(max_length=255, default="")
    avg_cost_gbp = models.DecimalField(max_digits=10, decimal_places=2, default=0)

    def __str__(self):
        return f"{self.part_name} - {self.make} {self.model} {self.year}"


class Problem(models.Model):
    problem_text = models.TextField()
    category = models.CharField(max_length=100)

    def __str__(self):
        return self.problem_text[:50]


class SavedCar(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="saved_cars")
    make = models.CharField(max_length=100)
    model = models.CharField(max_length=100)
    year = models.IntegerField()
    last_problem_text = models.TextField(blank=True, default="")
    last_diagnosis = models.CharField(max_length=100, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.make} {self.model} {self.year}"


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    is_mechanic = models.BooleanField(default=False)
    mechanic_code = models.CharField(max_length=100, blank=True, default="")

    def __str__(self):
        role = "Mechanic" if self.is_mechanic else "User"
        return f"{self.user.username} - {role}"


class DiagnosisRecord(models.Model):
    performed_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="diagnosis_records"
    )
    make = models.CharField(max_length=100)
    model = models.CharField(max_length=100)
    year = models.IntegerField()
    problem_text = models.TextField()
    diagnosis = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.make} {self.model} {self.year} - {self.diagnosis}"