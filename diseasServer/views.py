import json
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from MachineLearning.predict import trained_model_prediction, symptoms, Description

@csrf_exempt
def predict(request):
	if request.method == 'POST':
		data_json_str = request.body.decode('utf-8')
		# Convert the JSON string to a Python list
		data_list = json.loads(data_json_str)
		# print(data_list)
		if (len(data_list['Symptom']) == 0):
			object_error = {
				'condition' : 'you sould pass atleast one symptom',
				'description' : 'with no symptoms we can not make a diagnosis, thank you for your comprehension'
			}
			return JsonResponse(object_error, safe=False)
		prediction = trained_model_prediction(data_list)
		object_result = {
			'condition' : prediction,
			'description' : Description(prediction)[0]
		}
		return JsonResponse(object_result, safe=False)
	return JsonResponse(['Only POST requests are allowed'])

@csrf_exempt
def getSymptoms(request):
	if request.method == 'GET':
		result = symptoms()
		return JsonResponse(result, safe=False)
	return JsonResponse(['Only GET requests are allowed'])

@csrf_exempt
def getDescription(request, ds):
	if ds is None:
		return JsonResponse(['you should pass a condition as an argument'])
	if request.method == 'GET':
		result = Description(ds)
		return JsonResponse(result, safe=False)
	return JsonResponse(['Only POST requests are allowed'])

@csrf_exempt
def getfiltredData(request):
    if request.method == 'GET':
        data = pd.read_json("MachineLearning/symptoms.json", orient='index')
        data = data.T
        symptoms_dict = data.to_dict()
        for body_part in symptoms_dict:
            symptoms_dict[body_part] = list(symptoms_dict[body_part].values())
        return JsonResponse(symptoms_dict, safe=False)
    return JsonResponse(['Only POST requests are allowed'])
