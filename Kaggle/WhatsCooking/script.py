#!/usr/bin python
import json
import unicodedata
import numpy
import os.path
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

def getIngredientsAndCuisine( ):
	ingredientsFileName = 'Data/allIngredients.txt'
	cuisineFileName = 'Data/allCuisine.txt'
	allIngredients = [];
	allCuisine = [];
	if not os.path.exists(ingredientsFileName):
		with open('Data/train.json/train.json') as f:
			data = f.read()
			jsondata = json.loads(data)
		for row in jsondata:
			thisSetIngredients = row["ingredients"]
			for ingredient in thisSetIngredients:
				ingredient = ingredient.lower()
				detailedIngreds = ingredient.split()
				for dets in detailedIngreds:
					if dets not in allIngredients:
						allIngredients.append(dets)
	
		thefile = open('Data/allIngredients.txt','w')
		for ingredient in allIngredients:
			thefile.write(ingredient.encode('utf8'))
			thefile.write("\n")

		thefile.close()
	else:
		with open(ingredientsFileName) as f:
			allIngredients = f.read().decode('utf8').splitlines()
	
	if not os.path.exists(cuisineFileName):
		with open('Data/train.json/train.json') as f:
			data = f.read()
			jsondata = json.loads(data)
		for row in jsondata:
			thisCuisine = row["cuisine"]
			if thisCuisine not in allCuisine:
				allCuisine.append(thisCuisine)
	
		thefile = open('Data/allCuisine.txt','w')
		for cuisine in allCuisine:
			thefile.write(cuisine.encode('utf8'))
			thefile.write("\n")

		thefile.close()
	else:
		with open(cuisineFileName) as f:
			allCuisine = f.read().decode('utf8').splitlines()
	
	return [allIngredients,allCuisine]

def trainLinear(X,y):   # best 0.774
	print("Building and Training my classifier")
	myClasserLinear = OneVsRestClassifier(LinearSVC(random_state=0))
	myClasserLinear.fit(X,y)
	return[myClasserLinear]

def trainBayes(X,y):  # Best 0.72
	myClasserBayes = MultinomialNB()
	myClasserBayes.fit(X,y)
	return[myClasserBayes]
	
def trainSvm(X,y):		# best 0.19 :(
	print("Building and Training SVM classifier")
	myClasserSvm = svm.SVC(decision_function_shape='ova', kernel='sigmoid')
	myClasserSvm.fit(X,y)
	
def getPredictions(myClasser, XTest):
	print("Making predictions")
	predicted_cuisines = myClasser.predict(XTest)
	return [predicted_cuisines]

def makeTrainingSet(ingredientsList,cuisineList):
	X = []
	y = []
	with open('Data/train.json/train.json') as f:
		data = f.read()
		jsondata = json.loads(data)
		for row in jsondata:
			thisCuisine = row["cuisine"]
			thisSetIngredients = row["ingredients"]
			thisX = [0] * (len(ingredientsList) + 1)
			for ingredient in thisSetIngredients:
				ingredient = ingredient.lower()
				detailedIngreds = ingredient.split()
				for dets in detailedIngreds:
					thisIndex = ingredientsList.index(dets)
					thisX[thisIndex] = 1
			thisX[len(thisX)-1] = len(thisSetIngredients)
			X.append(thisX)
			y.append(cuisineList.index(thisCuisine))
			
	return[X,y]

def makeTestSet(ingredientsList,cuisineList):
	X = []
	testId = []
	with open('Data/test.json/test.json') as f:
		data = f.read()
		jsondata = json.loads(data)
		for row in jsondata:
			thisId = row["id"]
			thisSetIngredients = row["ingredients"]
			thisX = [0] * (len(ingredientsList) + 1)
			for ingredient in thisSetIngredients:
				ingredient = ingredient.lower()
				detailedIngreds = ingredient.split()
				for dets in detailedIngreds:
					if dets in ingredientsList:
						thisIndex = ingredientsList.index(dets)
						thisX[thisIndex] = 1
			thisX[len(thisX)-1] = len(thisSetIngredients)
			X.append(thisX)
			testId.append(thisId)
	return[X,testId]
	
def printResults(fileLocation, predicted_cuisine, testId, cuisineList):
	print("Predictions done, writing submission file")
	thefile = open(fileLocation,'w')
	thefile.write('id,cuisine')
	thefile.write('\n')

	for i in range(0,len(predicted_cuisine)):
		thefile.write(str(testId[i]))
		thefile.write(",")
		thefile.write(cuisineList[predicted_cuisine[i]].encode('utf8'))
		thefile.write("\n")
	thefile.close()
	return
################################# Program Starts Here #################
print("Step 1: Getting the Ingredients and Cuisine List")
[ingredientsList,cuisineList] = getIngredientsAndCuisine();

print("Step 2: Creating Training Set")
[trainX, trainY] = makeTrainingSet(ingredientsList,cuisineList)

print("Step 3: Training the Classifier")
[myClassifier] = trainLinear(trainX,trainY)
#[myClassifier] = trainBayes(trainX,trainY)

print("Step 4: Prepare the Test set")
[testX, testId] = makeTestSet(ingredientsList,cuisineList)

print("Step 5: Make predictions")
[thePredictions] = getPredictions(myClassifier, testX)

print("Step 6: Print Results")
printResults("Data/MySubmissionLin.csv", thePredictions, testId, cuisineList)

print("All done")

