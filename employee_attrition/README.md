## Employee Attiriton Prediciton
----------------------------------------------
### Overview

This project aims to predict employee attrition using a Logistic Regression model trained on various employee attributes. The model is deployed as a web application using Flask, providing an interactive UI for users to input employee details and obtain predictions.

### Tech Stack

**Machine Learning:** Python, Scikit-learn, Pandas, NumPy
**Web Framework:** Flask
**Frontend:** HTML, CSS, JavaScript (Jinja templates)
**Database (if needed):** SQLite / MongoDB (Optional for storing predictions)

### Installation & Setup

**Prerequisites**

    - Python 3.x
    - Flask
    - Scikit-learn
    - Pandas, NumPy

**Steps**

    1. Clone the repository
    ```
    git clone https://github.com/techiescamp/mlops.git
    cd mlops/employee_attrition
    ```

    2. Create a virtual environment (Optional but recommended)
    ```
    # for windows
    python -m venv venv
    source venv/bin/activate  # For macOS/Linux
    venv\Scripts\activate     # For Windows
    ```

    3. Install dependencies (if requirements.txt exists)
    ```
    pip install -r requirements.txt
    ```

    4. Train the Model
    ```
    cd employee_attrition_model
    python model.py
    ```

    This will generate my_model_lr.pkl in `employee_attrition_model/` directory.

    5. Run the Flask App
    ```
    # go back to employee_attrition/ directory
    cd ../

    # go to employee_attrition_ui/ directory
    cd employee_attrition_ui
    
    # run flask for ui
    python app.py
    ```

    6. Access the Web UI
    ``
    Open http://127.0.0.1:5000/ in your browser.
    ```

    7. Input employee details (e.g., Age, Monthly Income, Job Satisfaction, etc.).

    8. Click "Predict" to see the attrition result.

### Model Training

    - The dataset is split into training and testing sets.
    - Features are scaled using StandardScaler.
    - Categorical variables are encoded using OrdinalEncoder.
    - A Logistic Regression model is trained and saved as model.pkl.

### API Endpoints

Method              Endpoint          Description
----------------------------------------------------------------
  GET                   /            Load the homepage
 
  POST                 /predict      Predicts attrition based on input data

### Future Enhancements

- Improve UI/UX with better design.
- Deploy using Docker & cloud services.
- Use a more advanced model like Random Forest or XGBoost.

### License

This project is open-source and available under the MIT License.
**&copy; www.techiescamp.com/**