import pickle
import json

model = pickle.load(open('model_RF.pkl', 'rb'))

data = ['Dollar_Amount', 'Transaction_Type', 'Store_Type', 'Cardholder_Region',
       'Country', 'Hours_Since_Last_Transaction',
       'Total_Amount_of_Transaction_in_1_Day', '#Of_Transaction_in_1Day',
       'Avg._Per_Transaction', 'Trans_Weekend', 'Trans_Night']

list = []

def predict(args):
  for key in data:
    value = float(args.get(key))
    # petal_width = float(args.get("petal_width"))
    list.append(value)

  newdata = list
  classPrediction = model.predict([newdata])
  score_prediction = model.predict_proba([newdata])

  classname = classPrediction[0]
  
  if classname == 0:
    scorePrediction = score_prediction[0][0]
  else:
    scorePrediction = score_prediction[0][1]

  result = {
    "className" : classname.tolist(),
    "scorePrediction": scorePrediction.tolist()
  }
  
  return json.dumps(result)

