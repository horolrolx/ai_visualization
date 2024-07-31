from django.shortcuts import render
from django.http import HttpResponse

# Create your view here.
def index(request):
    return HttpResponse('<h1>My first app</h1>')

def myfirst(request):
    return render(request, "myfirst.html")

def classification(request):
    return render(request, "classification.html")