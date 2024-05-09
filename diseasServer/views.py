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
		if (len(data_list['Symptom']) == 0):
			object_error = {
				'condition' : 'you sould pass atleast one symptom',
				'description' : 'with no symptoms we can not make a diagnosis, thank you for your comprehension'
			}
			return JsonResponse(object_error, safe=False)
		prediction = trained_model_prediction(data_list['Symptom'])
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
	if request.method == 'POST':
		if ds is None:
			return JsonResponse(['you should pass a condition as an argument'])
		result = Description(ds)
		return JsonResponse(result, safe=False)
	return JsonResponse(['Only POST requests are allowed'])

@csrf_exempt
def getfiltredData(request):
	if request.method == 'GET' || request.method == 'POST':
		data = pd.read_json("MachineLearning/symptoms.json", orient='index')
		data = data.T
		symptoms_dict = data.to_dict()
		send = {}
		for key, value in symptoms_dict.items():
			value = list(value.values())
			send[key] = [item for item in value if item is not None]
		return JsonResponse(send, safe=False)
	return JsonResponse(['Only GET or POST requests are allowed'])
