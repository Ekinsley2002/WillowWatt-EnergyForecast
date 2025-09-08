# WillowWatt-EnergyForecast

## Current Directory Layout
- **`Data/`**  
  Holds both live and historic datasets.  
  *Currently: simple one-year historic dataset.*

- **`DataAnalysis/`**  
  Sandbox for one-off testing scripts (data collection, exploration, quick experiments).

- **`ReferenceCode/`**  
  Stores prior coding logic.  
  *Try to trim/comment out anything we don’t plan to reuse.*

- **`main.py`**  
  Entry point for forecasting.  
  *Currently: a simple procedure; later it will orchestrate live data collection, learning, and model calls.*

- **`requirements.txt`**  
  Dependencies for the project.  
  ```bash
  pip install -r requirements.txt


This is what the structure should look like once everything is built:
WillowWatt-EnergyForecast/


```plaintext
├── Data/                 
│   ├── historic/          
│   └── live/              
│
├── DataAnalysis/          
│   ├── preprocessing/     
│   └── experiments/      
│
├── ReferenceCode/       
│   └── legacy/          
│
├── src/               
│   ├── models/          
│   ├── utils/      
│   └── pipeline/       
│
├── main.py              
├── requirements.txt      
└── README.md


Submitting something new:
#  **Submitting Something New**

If you would like to fix an issue or make a system better, please:

1. **Submit an issue** on the GitHub issues page describing how you plan to fix the problem.  
   *(For records — it will likely be approved quickly.)*

2. **Create a new branch** named:  
 "your_first_name_ - what you plan to fix "

3. **Implement your changes** off `main`.

4. **Request a merge (Pull Request)** back into `main` so changes can be reviewed.

5. **Delete your branch** once your changes are merged. 



