from pathlib import Path
import json
import joblib

from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.models import Count
from django.shortcuts import redirect, render

from .forms import (
    CarDetailsForm,
    RegisterForm,
    LoginForm,
    MechanicRegisterForm,
    MechanicLoginForm,
)
from .models import Part, SavedCar, UserProfile, DiagnosisRecord

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "problem_classifier.joblib"


def get_or_create_profile(user):
    profile, _ = UserProfile.objects.get_or_create(user=user)
    return profile


def register(request):
    if request.user.is_authenticated:
        return redirect("diagnose")

    form = RegisterForm(request.POST or None)

    if request.method == "POST" and form.is_valid():
        email = form.cleaned_data["email"].strip().lower()
        password = form.cleaned_data["password"]

        user = User.objects.create_user(
            username=email,
            email=email,
            password=password
        )

        profile = get_or_create_profile(user)
        profile.is_mechanic = False
        profile.mechanic_code = ""
        profile.save()

        login(request, user)
        return redirect("diagnose")

    return render(request, "main/register.html", {"form": form})


def pricing(request):
    return render(request, "main/pricing.html")


def contact(request):
    return render(request, "main/contact.html")


def login_view(request):
    if request.user.is_authenticated:
        return redirect("diagnose")

    form = LoginForm(request.POST or None)
    error_message = None

    if request.method == "POST" and form.is_valid():
        email = form.cleaned_data["email"].strip().lower()
        password = form.cleaned_data["password"]

        user = authenticate(request, username=email, password=password)
        if user is not None:
            profile = get_or_create_profile(user)

            if profile.is_mechanic:
                error_message = "This is a mechanic account. Please use the MechaniX login."
            else:
                login(request, user)
                next_url = request.GET.get("next")
                if next_url:
                    return redirect(next_url)
                return redirect("diagnose")
        else:
            error_message = "Invalid email or password."

    return render(
        request,
        "main/login.html",
        {
            "form": form,
            "error_message": error_message
        }
    )


def mechanic_register(request):
    if request.user.is_authenticated:
        profile = get_or_create_profile(request.user)
        if profile.is_mechanic:
            return redirect("mechanix_dashboard")

    form = MechanicRegisterForm(request.POST or None)

    if request.method == "POST" and form.is_valid():
        email = form.cleaned_data["email"].strip().lower()
        code = form.cleaned_data["code"].strip()
        password = form.cleaned_data["password"]

        user = User.objects.create_user(
            username=email,
            email=email,
            password=password
        )

        profile = get_or_create_profile(user)
        profile.is_mechanic = True
        profile.mechanic_code = code
        profile.save()

        login(request, user)
        return redirect("mechanix_dashboard")

    return render(request, "main/mechanic_register.html", {"form": form})


def mechanic_login(request):
    if request.user.is_authenticated:
        profile = get_or_create_profile(request.user)
        if profile.is_mechanic:
            return redirect("mechanix_dashboard")

    form = MechanicLoginForm(request.POST or None)
    error_message = None

    if request.method == "POST" and form.is_valid():
        email = form.cleaned_data["email"].strip().lower()
        code = form.cleaned_data["code"].strip()
        password = form.cleaned_data["password"]

        user = authenticate(request, username=email, password=password)
        if user is not None:
            profile = get_or_create_profile(user)

            if not profile.is_mechanic:
                error_message = "This account is not registered as a mechanic."
            elif profile.mechanic_code != code:
                error_message = "Invalid mechanic code."
            else:
                login(request, user)
                return redirect("mechanix_dashboard")
        else:
            error_message = "Invalid email, code, or password."

    return render(
        request,
        "main/mechanic_login.html",
        {
            "form": form,
            "error_message": error_message
        }
    )


def logout_view(request):
    logout(request)
    return redirect("diagnose")


def mechanix_entry(request):
    if request.user.is_authenticated:
        profile = get_or_create_profile(request.user)
        if profile.is_mechanic:
            return redirect("mechanix_dashboard")
    return redirect("mechanic_login")


@login_required(login_url="mechanic_login")
def mechanix_dashboard(request):
    profile = get_or_create_profile(request.user)
    if not profile.is_mechanic:
        return redirect("mechanic_login")

    records = DiagnosisRecord.objects.filter(performed_by=request.user).order_by("-created_at")

    grouped = (
        records.values("diagnosis")
        .annotate(total=Count("id"))
        .order_by("-total")
    )

    max_total = max([item["total"] for item in grouped], default=1)

    chart_data = []
    for item in grouped:
        width = int((item["total"] / max_total) * 100) if max_total else 0
        chart_data.append({
            "category": item["diagnosis"],
            "total": item["total"],
            "width": width,
        })

    total_jobs = records.count()
    top_category = chart_data[0]["category"].upper() if chart_data else "No data yet"

    return render(
        request,
        "main/mechanix.html",
        {
            "chart_data": chart_data,
            "total_jobs": total_jobs,
            "top_category": top_category,
            "recent_records": records[:8],
        }
    )


@login_required(login_url="login")
def my_cars(request):
    cars = SavedCar.objects.filter(user=request.user).order_by("-created_at")
    return render(request, "main/my_cars.html", {"cars": cars})


@login_required(login_url="login")
def save_car(request):
    if request.method == "POST":
        make = request.POST.get("make", "").strip()
        model_name = request.POST.get("model", "").strip()
        year = request.POST.get("year", "").strip()
        problem_text = request.POST.get("problem_text", "").strip()
        diagnosis = request.POST.get("diagnosis", "").strip()

        if make and model_name and year.isdigit():
            saved_car = SavedCar.objects.filter(
                user=request.user,
                make__iexact=make,
                model__iexact=model_name,
                year=int(year),
            ).first()

            if saved_car:
                saved_car.last_problem_text = problem_text
                saved_car.last_diagnosis = diagnosis
                saved_car.save()
            else:
                SavedCar.objects.create(
                    user=request.user,
                    make=make,
                    model=model_name,
                    year=int(year),
                    last_problem_text=problem_text,
                    last_diagnosis=diagnosis,
                )

    return redirect("my_cars")


def diagnose_problem(request):
    form = CarDetailsForm(request.POST or None)
    diagnosis = None
    recommended_parts = None
    model_missing = False
    form_error = None

    parts = Part.objects.all().values("make", "model", "year")
    vehicle_map = {}

    for row in parts:
        make = row["make"]
        model = row["model"]
        year = row["year"]

        if make not in vehicle_map:
            vehicle_map[make] = {}

        if model not in vehicle_map[make]:
            vehicle_map[make][model] = []

        if year not in vehicle_map[make][model]:
            vehicle_map[make][model].append(year)

    for make in vehicle_map:
        for model_name in vehicle_map[make]:
            vehicle_map[make][model_name] = sorted(vehicle_map[make][model_name])

    selected_make = request.POST.get("make", "")
    selected_model = request.POST.get("model", "")
    selected_year = request.POST.get("year", "")

    if request.method == "POST" and form.is_valid():
        make = form.cleaned_data["make"].strip()
        model_name = form.cleaned_data["model"].strip()
        year = form.cleaned_data["year"].strip()
        problem_text = form.cleaned_data["problem_text"].strip().lower()

        if make not in vehicle_map:
            form_error = "Selected make is not available."
        elif model_name not in vehicle_map[make]:
            form_error = "Selected model is not available for that make."
        elif not year.isdigit() or int(year) not in vehicle_map[make][model_name]:
            form_error = "Selected year is not available for that make and model."
        elif not MODEL_PATH.exists():
            model_missing = True
        else:
            model = joblib.load(MODEL_PATH)

            probabilities = model.predict_proba([problem_text])[0]
            classes = model.classes_

            best_index = probabilities.argmax()
            best_probability = probabilities[best_index]
            predicted_category = classes[best_index]

            if best_probability < 0.55:
                form_error = "We could not confidently understand that issue. Please describe the symptom more clearly."
            else:
                diagnosis = predicted_category
                recommended_parts = Part.objects.filter(
                    category=diagnosis,
                    make__iexact=make,
                    model__iexact=model_name,
                    year=int(year),
                )

                DiagnosisRecord.objects.create(
                    performed_by=request.user if request.user.is_authenticated else None,
                    make=make,
                    model=model_name,
                    year=int(year),
                    problem_text=problem_text,
                    diagnosis=diagnosis,
                )

    return render(
        request,
        "main/diagnosis.html",
        {
            "form": form,
            "diagnosis": diagnosis,
            "recommended_parts": recommended_parts,
            "model_missing": model_missing,
            "form_error": form_error,
            "vehicle_map_json": json.dumps(vehicle_map),
            "selected_make": selected_make,
            "selected_model": selected_model,
            "selected_year": selected_year,
        },
    )