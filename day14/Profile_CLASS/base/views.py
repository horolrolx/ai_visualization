from django.shortcuts import render

# Create your views here.
def home(request):
    return render(request, "home.html")

def projects(request):
    return render(request, "projects.html")

def contact(request):
    return render(request, "contact.html")

def mnist(request):
    file = request.FILES.get['image']
    if not file:
        return render(request, "mnist.html")
    else:
        return render(request, "mnist.html"), {"letter": letter, "probs": probs})