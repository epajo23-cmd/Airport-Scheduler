# Intelligent Airport Runway Scheduling Agent

## Group Members

- Thomas Kroj
- Eden Pajo

---

## Project Overview

This project implements an **intelligent agent** for airport runway scheduling as part of the CEN 352 term project.

The agent integrates **two different AI techniques** taught in the course:

1. **Statistical Learning (Machine Learning)**  
   A classification model is trained on historical flight data to predict the probability that a flight will experience a **departure delay greater than 15 minutes**.

2. **Planning / Scheduling**  
   The predicted delay risk is used to make rational scheduling decisions by determining the order in which flights are assigned to a single runway.  
   The intelligent scheduling strategy is compared against a **First-Come-First-Served (FCFS)** baseline.

---

## AI Approach Summary

- **Sensors:**  
  Flight attributes (airline, origin airport, destination airport, scheduled departure time, day information, distance) and the current flight queue.

- **Decision Process:**  
  A machine learning classifier predicts delay risk for each flight.  
  A scheduling policy prioritizes flights with higher predicted delay risk.

- **Actuators:**  
  Selection of the flight ordering for runway departure.

---

## Installation

Install the required Python dependencies:

pip install -r requirements.txt

Dataset
The project uses the Flight Delays and Cancellations (2015) dataset from Kaggle.
https://www.kaggle.com/datasets/usdot/flight-delays?select=flights.csv

The dataset file flights.csv should be added in the repository under:
data/flights.csv

Running the Project

Train the model
python model/train.py

Evaluate the model
python model/evaluate.py

Run the Streamlit application
streamlit run app.py

The Streamlit interface allows interactive comparison between FCFS scheduling and the intelligent scheduling agent.
