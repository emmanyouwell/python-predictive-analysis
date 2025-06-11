import pandas as pd
import pymongo
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

from oauth2client.service_account import ServiceAccountCredentials
import gspread
import joblib
from dotenv import load_dotenv
import os
import base64
# Load .env from one level up since the script is in /app
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))



# --- CONFIGURATION ---
# Now use the variables
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MODEL_PATH = os.getenv("MODEL_PATH")
SPREADSHEET_NAME = os.getenv("SPREADSHEET_NAME")
# Path where the creds file will be written
CREDENTIALS_FILE = "creds/creds.json"

# Decode and write the credentials file if not already present
if not os.path.exists(CREDENTIALS_FILE):
    encoded = os.getenv("GOOGLE_CREDS_B64")
    if not encoded:
        raise RuntimeError("Missing GOOGLE_CREDS_B64 environment variable")
    
    os.makedirs(os.path.dirname(CREDENTIALS_FILE), exist_ok=True)
    with open(CREDENTIALS_FILE, "wb") as f:
        f.write(base64.b64decode(encoded))

# --- GOOGLE SHEET COLUMN MAPPING ---
columns_to_change = [
    "Have you donated breast milk before? (Nakapagbigay ka na ba ng iyong gatas dati?)",
    "Will you be allowed by your husband to donate your breast milk to the Taguig City Human Milk Bank? (Papayagan ka ba ng iyong asawa na magbigay ng iyong gatas sa Taguig City Human Milk Bank?)",
    "Have you for any reason been deferred as a milk donor? (Ikaw ba ay natangihan na magbigay ng iyong gatas / breastmilk?)",
    "Did you have a normal pregnancy and delivery for your most recent pregnancy? (Normal ba ang panganganak mo sa huli mong anak?)",
    "Do you have any acute or chronic infection, systemic disorders, tuberculosis or history of hepatitis? (Mayroon ka bang kahit anong impeksyon o sakit? Nagkaroon ng sakit sa atay dati?)", 
    "Have you received any blood transfusion or other blood products within the last 12 months? (Ikaw ba ay nasalinan ng dugo o kahit anong produkto mula sa dugo nitong nakaraang 12 buwan?)",
    "Have you received any organ or tissue transplant within the last 12 months? (Ikaw ba ay nakatanggap ng parte ng katawan mula sa ibang tao nitong nakaraang 12 buwan?)",
    "Within the last 24 hours, have you had intake of any hard liquor or alcohol? (Nakainom ka ba ng alak nitong nakaraang 24 oras?)",
    "Do you regularly use over-the-counter medications or systemic preparations such as replacement hormones and some birth control hormones? (Regular ka bang gumagamit ng mga gamot gaya ng mga hormones o pills?)",
    "Do you use megadose vitamins or pharmacologically active herbal preparations? (Gumagamit ka ba ng mga \"megadose vitamins\" o mga \"herbal drugs\"?)",
    "Are you a total vegetarian/vegan? (Ikaw ba ay hindi kumakain ng karne o isang vegetarian?)",
    "Have you had breast augmentation surgery, using silicone breast implants? (Ikaw ba ay naoperahan na sa suso at nalagyan ng “silicone” o artipisyal na breast implants?)",
    "Do you use illicit drugs? (Gumagamit ka ba ng ipinagbabawal na gamot?)",
    "Do you smoke? (Ikaw ba ay naninigarilyo?)",
    "Have you had syphilis, HIV, herpes, or any sexually-transmitted disease? (Nagkakaroon ka ba ng sakit na nakukuha sa pakikipagtalik /sex?)",
    "Do you have multiple sex partners? (Nagkaroon ka ba ng karanasang makipagtalik sa hindi lang iisang lalaki?)",
    "Have you had a sexual partner from any of the following? (Nagkaroon ka ba ng partner mula sa mga sumusunod?)",
    "Have you had a tattoo applied or have had accidental needlestick or contact with someone else's blood? (Nagpalagay ka na ba ng tattoo, naturukan ng karayom nang hindi sinasadya o nadikit sa dugo ng ibang tao?)",
    "Is your child healthy? (Malusog ba ang iyong anak?)",
    "Was your child delivered full term? (Ipinanganak ba ang anak mo na husto sa buwan?)",
    "Are you exclusively breastfeeding your child? (Purong gatas mo ba ang binibigay mo sa anak mo at walang halong ibang formula / gatas?)",
    "Is/Was your youngest child jaundiced? (Madilaw/nanilaw ba ang bunso mong anak?)",
    "Has your child ever received milk from another mother? (Nakatanggap na ba ang iyong anak ng gatas/ breast milk mula sa ibang ina?)",
]

# --- FASTAPI SETUP ---
app = FastAPI()

class PredictRequest(BaseModel):
    submissionID: str

# --- LOAD GOOGLE SHEET ---
def load_google_sheet():
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ])
    client = gspread.authorize(creds)
    sheet = client.open(SPREADSHEET_NAME).sheet1
    df = pd.DataFrame(sheet.get_all_records())
    return df

# --- LOAD LABELS FROM MONGODB ---
def load_labels():
    client = pymongo.MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]
    docs = list(collection.find(
        {
            "submissionID": {"$exists": True, "$ne": None},
            "eligibility": {"$exists": True, "$ne": None}
        },
        {"_id": 0, "submissionID": 1, "eligibility": 1}
    ))
    return pd.DataFrame(docs)

# --- MERGE, CLEAN, TRAIN ---
def prepare_data():
    sheet_df = load_google_sheet().rename(columns={"Submission ID": "submissionID"})
    label_df = load_labels()

    print(f"sheet_df submissionIDs: {sheet_df['submissionID'].unique()}")
    print(f"label_df submissionIDs: {label_df['submissionID'].unique()}")

    df = sheet_df.merge(label_df, on="submissionID")
    print(f"Merged df after join: {len(df)} rows")
    
    for col in columns_to_change:
        df[col] = df[col].astype(str).str.strip().str.lower().map({
            "yes": 1,
            "no": 0,
            "none of the above": 0
        })
    print(f"df after mapping: {len(df)} rows")
    df = df.dropna(subset=columns_to_change + ["eligibility"])
    print(f"df after dropping NAs: {len(df)} rows")
    df["eligibility"] = df["eligibility"].map({"Eligible": 1, "Ineligible": 0})
    print(f"Full df: {len(df)} rows")
    return df


def train_and_evaluate():
    try:
        df = prepare_data()
        print("Data preparation successful.")
    except Exception as e:
        print(f"[prepare_data ERROR]: {e}")
        raise
   
   
    if df.empty:
        raise ValueError("No data available for training. Check Google Sheet and MongoDB labels.")

    X = df[columns_to_change]
    y = df["eligibility"]
    print(f"Total rows after cleaning: {len(df)}")
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
       
        # Training with no validation set    
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)

        
    except ValueError as e:
        print(f"[train_test_split ERROR]: {e}")
        raise

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, target_names=["Ineligible", "Eligible"], output_dict=True)
    print("Classification Report:", report)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    return acc, cm, report
def get_model_report():
    try:
        df = prepare_data()
        print("Data preparation successful.")
    except Exception as e:
        print(f"[prepare_data ERROR]: {e}")
        raise
   
   
    if df.empty:
        raise ValueError("No data available for training. Check Google Sheet and MongoDB labels.")
    model = joblib.load(MODEL_PATH)
    X = df[columns_to_change]
    y = df["eligibility"]
    print(f"Total rows after cleaning: {len(df)}")
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
       
        # Training with no validation set    
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
    
        
    except ValueError as e:
        print(f"[train_test_split ERROR]: {e}")
        raise
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, target_names=["Ineligible", "Eligible"], output_dict=True)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    return acc, cm, report

@app.on_event("startup")
def startup_event():
    if not os.path.exists(MODEL_PATH):
        train_and_evaluate()

@app.post("/predict")
def predict(request: PredictRequest):
    acc, cm, report = train_and_evaluate()
    model = joblib.load(MODEL_PATH)
    df = load_google_sheet()
    df = df.rename(columns={"Submission ID": "submissionID"})
    row = df[df["submissionID"] == request.submissionID]
    if row.empty:
        raise HTTPException(status_code=404, detail="Submission ID not found")
    try:
        features = row.iloc[0][columns_to_change].map({"Yes": 1, "No": 0})
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Missing column in survey response: {e}. Check if all expected questions are present."
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing survey answers: {str(e)}"
        )
    
    prediction = model.predict([features])[0]
    
  
    return {
        "prediction": "Eligible" if prediction == 1 else "Ineligible",
        "accuracy": f"{acc:.2%}",
        "confusion_matrix": cm.tolist(),
        "precision": {
            "Eligible": f"{report['Eligible']['precision']:.2f}",
            "Ineligible": f"{report['Ineligible']['precision']:.2f}"
        },
        "recall": {
            "Eligible": f"{report['Eligible']['recall']:.2f}",
            "Ineligible": f"{report['Ineligible']['recall']:.2f}"
        },
        "f1_score": {
            "Eligible": f"{report['Eligible']['f1-score']:.2f}",
            "Ineligible": f"{report['Ineligible']['f1-score']:.2f}"
        }
    }

@app.get("/retrain")
def retrain():
    # train_and_evaluate()
    acc, cm, report = train_and_evaluate()
    return {
        "message": "Model retrained",
        "accuracy": f"{acc:.2%}",
        "confusion_matrix": cm.tolist(),
        "precision": {
            "Eligible": f"{report['Eligible']['precision']:.2f}",
            "Ineligible": f"{report['Ineligible']['precision']:.2f}"
        },
        "recall": {
            "Eligible": f"{report['Eligible']['recall']:.2f}",
            "Ineligible": f"{report['Ineligible']['recall']:.2f}"
        },
        "f1_score": {
            "Eligible": f"{report['Eligible']['f1-score']:.2f}",
            "Ineligible": f"{report['Ineligible']['f1-score']:.2f}"
        }
    }

@app.get("/model_report")
def model_report():
    try:
        acc, cm, report = get_model_report()
        return {
            "message": "Model report generated",
            "accuracy": f"{acc:.2%}",
            "confusion_matrix": cm.tolist(),
            "precision": {
                "Eligible": f"{report['Eligible']['precision']:.2f}",
                "Ineligible": f"{report['Ineligible']['precision']:.2f}"
            },
            "recall": {
                "Eligible": f"{report['Eligible']['recall']:.2f}",
                "Ineligible": f"{report['Ineligible']['recall']:.2f}"
            },
            "f1_score": {
                "Eligible": f"{report['Eligible']['f1-score']:.2f}",
                "Ineligible": f"{report['Ineligible']['f1-score']:.2f}"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating model report: {str(e)}")