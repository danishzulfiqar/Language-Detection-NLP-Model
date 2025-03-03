# importing libraries
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
import sklearn
import pickle
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

# Initialize FastAPI app
app = FastAPI()

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize LabelEncoder
le = LabelEncoder()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    # loading the dataset
    data = pd.read_csv("language_detection.csv")
    y = data["Language"]

    # label encoding
    y = le.fit_transform(y)

    # loading the model and cv
    model = pickle.load(open("model.pkl", "rb"))
    cv = pickle.load(open("transform.pkl", "rb"))

    # preprocessing the text
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', '', text)
    text = re.sub(r'[[]]', '', text)
    text = text.lower()
    dat = [text]
    
    # creating the vector
    vect = cv.transform(dat).toarray()
    
    # prediction
    my_pred = model.predict(vect)
    my_pred = le.inverse_transform(my_pred)

    return templates.TemplateResponse(
        "home.html", 
        {
            "request": request,
            "pred": f"The above text is in {my_pred[0]}"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)