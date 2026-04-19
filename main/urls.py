from django.urls import path
from . import views

urlpatterns = [
    path("", views.diagnose_problem, name="diagnose"),
    path("diagnose/", views.diagnose_problem, name="diagnose"),

    path("register/", views.register, name="register"),
    path("login/", views.login_view, name="login"),

    path("mechanix/", views.mechanix_entry, name="mechanix_entry"),
    path("mechanix/login/", views.mechanic_login, name="mechanic_login"),
    path("mechanix/register/", views.mechanic_register, name="mechanic_register"),
    path("mechanix/dashboard/", views.mechanix_dashboard, name="mechanix_dashboard"),

    path("logout/", views.logout_view, name="logout"),
    path("my-cars/", views.my_cars, name="my_cars"),
    path("save-car/", views.save_car, name="save_car"),

    path("pricing/", views.pricing, name="pricing"),
    path("contact/", views.contact, name="contact"),
]