from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

def index(request):
    return render(request, "index.html")

@csrf_exempt
def result(request):
    name = request.POST.get["name"]
    height = int(request.POST.get["height"])
    weight = int(request.POST.get["weight"])
    bmi = round(((weight / height) / height) * 10000, 2)
    context = {
        'name': name,
        'hegiht': height,
        'weight': weight,
        'bmi': bmi
    }
    return render(request, 'result.html', context)