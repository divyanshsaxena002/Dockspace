# Titanic Survival Predictor 🚢💡

Welcome to the **Titanic Survival Predictor**! This project allows users to predict whether a passenger survived or not on the Titanic, based on key factors like **age**, **sex**, **fare**, and more. 🎯

## 🚀 Features:
- Predict Titanic survival chances based on user inputs such as **Pclass**, **Age**, **Sex**, and more!
- Simple and user-friendly interface powered by **Streamlit**. 📊
- Containerized for easy deployment using **Docker**. 📦

---

## 🌅 Preview

Here’s a preview of the Titanic Survival Predictor app:

![App Preview](https://github.com/Tanmay-hue/Streamlit_Titanic_Survival/blob/master/image.png)

---

## 🌟 How It Works:
1. **Data Preprocessing:** The dataset is cleaned, missing values are handled, and categorical variables are converted into numerical values.
2. **Machine Learning Model:** A **RandomForestClassifier** is trained to predict survival using the processed features.
3. **Streamlit Interface:** Users can interact with the app, input their data, and instantly get survival predictions!

---

## ⚙️ Running Locally

### Step 1: Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/tanmay-hue/Streamlit_Titanic_Survival.git
cd Streamlit_Titanic_Survival
```

### Step 2: Install Dependencies

Install all the necessary Python libraries using `pip`:

```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit App

To start the Titanic Survival Predictor locally, run:

```bash
streamlit run app.py
```

You should see the app running on `http://localhost:8501`.

---

## 🐳 Docker Setup

### Step 1: Build the Docker Image

To run this app inside a Docker container, build the Docker image:

```bash
docker build -t titanic-survival-predictor .
```

### Step 2: Run the Docker Container

Once the image is built, run the container:

```bash
docker run -p 8501:8501 titanic-survival-predictor
```

This will run the app on `http://localhost:8501`, just like running it locally!

---

## 🚀 Deploying on GitHub Pages

1. Push your code to GitHub:

```bash
git add .
git commit -m "Deploy Streamlit_Titanic_Survival with Docker"
git push origin master
```

2. Visit your GitHub repository to see the deployed app!

---

## 🔍 Project Structure

```
titanic-survival-predictor/
│
├── app.py              # Streamlit app code
├── Dockerfile          # Docker configuration
├── requirements.txt    # List of dependencies
├── titanic_model.pkl   # Trained machine learning model
└── titanic_model.py    # Model training script (optional)
```

---

## 🔧 Technologies Used

- **Python** (for data science and machine learning)
- **Streamlit** (for building the interactive web app)
- **RandomForestClassifier** (for the machine learning model)
- **Docker** (for containerizing the app)
- **GitHub** (for version control and deployment)

---

Happy coding and may the odds of survival be ever in your favor! 🏆💻

By - Tanmay Singh
```
