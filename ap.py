import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd

# Set page config
# The correct function is st.set_page_config()
st.set_page_config(
    page_title="Employee Attrition Prediction",
    layout="centered"
)

# Load the trained model
# The model file should be in the same directory as the script or a known path.
# Using a relative path is more robust than an absolute path like 'C:\Users\ADMIN...'.
MODEL_PATH = r'C:\Users\ADMIN\Downloads\model(rf).pkl'

@st.cache_resource
def load_model(path):
    """Loads the trained model from the specified path."""
    if not os.path.exists(path):
        st.error(f"Error: Model file not found at '{path}'. Please ensure the 'model(rf).pkl' file is in the same directory as this script.")
        return None
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model(MODEL_PATH)

# Define feature names (must match model training order)
# Moved to a more central location for clarity.
FEATURES = ['Age', 'Department', 'DistanceFromHome', 'EducationField',
            'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobRole',
            'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate',
            'NumCompaniesWorked', 'OverTime', 'StockOptionLevel',
            'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
            'YearsSinceLastPromotion', 'YearsWithCurrManager']

# Corrected and consolidated option mappings
DEPARTMENT_OPTIONS = {
    "Sales": 0, "Research & Development": 1, "Human Resources": 2
}
EDUCATION_FIELD_OPTIONS = {
    "Life Sciences": 0, "Other": 1, "Medical": 2, "Marketing": 3, "Technical Degree": 4, "Human Resources": 5
}
JOB_ROLE_OPTIONS = {
    "Sales Executive": 0, "Research Scientist": 1, "Laboratory Technician": 2,
    "Manufacturing Director": 3, "Healthcare Representative": 4, "Manager": 5,
    "Sales Representative": 6, "Research Director": 7, "Human Resources": 8
}
MARITAL_STATUS_OPTIONS = {
    "Single": 0, "Married": 1, "Divorced": 2
}
OVERTIME_OPTIONS = {
    "No": 0, "Yes": 1
}

# Invert mappings for display purposes
INV_DEPARTMENT_OPTIONS = {v: k for k, v in DEPARTMENT_OPTIONS.items()}
INV_EDUCATION_FIELD_OPTIONS = {v: k for k, v in EDUCATION_FIELD_OPTIONS.items()}
INV_JOB_ROLE_OPTIONS = {v: k for k, v in JOB_ROLE_OPTIONS.items()}
INV_MARITAL_STATUS_OPTIONS = {v: k for k, v in MARITAL_STATUS_OPTIONS.items()}
INV_OVERTIME_OPTIONS = {v: k for k, v in OVERTIME_OPTIONS.items()}

# Sample data for new features
sample_employees = [
    {
        'Name': 'Alice', 'Age': 35, 'Department': 0, 'DistanceFromHome': 2, 'EducationField': 2,
        'EnvironmentSatisfaction': 4, 'JobInvolvement': 4, 'JobLevel': 3, 'JobRole': 0,
        'JobSatisfaction': 4, 'MaritalStatus': 1, 'MonthlyIncome': 9000, 'MonthlyRate': 15000,
        'NumCompaniesWorked': 1, 'OverTime': 0, 'StockOptionLevel': 1,
        'TotalWorkingYears': 10, 'YearsAtCompany': 10, 'YearsInCurrentRole': 8,
        'YearsSinceLastPromotion': 5, 'YearsWithCurrManager': 9
    },
    {
        'Name': 'Bob', 'Age': 28, 'Department': 1, 'DistanceFromHome': 15, 'EducationField': 0,
        'EnvironmentSatisfaction': 1, 'JobInvolvement': 2, 'JobLevel': 1, 'JobRole': 2,
        'JobSatisfaction': 1, 'MaritalStatus': 0, 'MonthlyIncome': 2500, 'MonthlyRate': 5000,
        'NumCompaniesWorked': 4, 'OverTime': 1, 'StockOptionLevel': 0,
        'TotalWorkingYears': 5, 'YearsAtCompany': 2, 'YearsInCurrentRole': 1,
        'YearsSinceLastPromotion': 0, 'YearsWithCurrManager': 1
    },
    {
        'Name': 'Charlie', 'Age': 42, 'Department': 1, 'DistanceFromHome': 8, 'EducationField': 2,
        'EnvironmentSatisfaction': 3, 'JobInvolvement': 3, 'JobLevel': 4, 'JobRole': 7,
        'JobSatisfaction': 3, 'MaritalStatus': 2, 'MonthlyIncome': 15000, 'MonthlyRate': 20000,
        'NumCompaniesWorked': 2, 'OverTime': 0, 'StockOptionLevel': 2,
        'TotalWorkingYears': 20, 'YearsAtCompany': 15, 'YearsInCurrentRole': 10,
        'YearsSinceLastPromotion': 7, 'YearsWithCurrManager': 10
    }
]

def get_employee_features(employee_dict):
    """Converts a dictionary of employee data into a list of features for the model."""
    return [
        employee_dict['Age'], employee_dict['Department'], employee_dict['DistanceFromHome'], employee_dict['EducationField'],
        employee_dict['EnvironmentSatisfaction'], employee_dict['JobInvolvement'], employee_dict['JobLevel'], employee_dict['JobRole'],
        employee_dict['JobSatisfaction'], employee_dict['MaritalStatus'], employee_dict['MonthlyIncome'], employee_dict['MonthlyRate'],
        employee_dict['NumCompaniesWorked'], employee_dict['OverTime'], employee_dict['StockOptionLevel'],
        employee_dict['TotalWorkingYears'], employee_dict['YearsAtCompany'], employee_dict['YearsInCurrentRole'],
        employee_dict['YearsSinceLastPromotion'], employee_dict['YearsWithCurrManager']
    ]

def add_new_features():
    """Adds the expanded features and insights to the Streamlit app."""
    st.markdown("---")
    
    # Feature 2: High-Risk Employee List
    with st.expander("High-Risk Employee List"):
        st.subheader("Employees with High Attrition Risk")
        high_risk_employees = []
        if model is not None:
            for emp in sample_employees:
                user_input = get_employee_features(emp)
                input_array = np.array(user_input).reshape(1, -1)
                try:
                    # Model expects 2D array
                    probability = model.predict_proba(input_array)[0]
                    attrition_prob = probability[1]
                    if attrition_prob > 0.5:
                        emp_copy = emp.copy()
                        emp_copy['Attrition Probability'] = f"{attrition_prob:.2f}"
                        high_risk_employees.append(emp_copy)
                except Exception as e:
                    st.warning(f"Could not predict for {emp['Name']}: {str(e)}")
            
            if high_risk_employees:
                # Create a DataFrame for a clean table display
                df = pd.DataFrame(high_risk_employees)
                st.table(df[['Name', 'Attrition Probability', 'OverTime', 'JobSatisfaction', 'MonthlyIncome']])
            else:
                st.info("No employees identified as high-risk in the sample list.")

    # Feature 3: Job Satisfaction & Performance Insights
    with st.expander("Job Satisfaction & Performance Insights"):
        st.subheader("High Job Satisfaction & Performance")
        high_performers = [emp for emp in sample_employees if emp['JobSatisfaction'] >= 3 and emp['JobInvolvement'] >= 3]
        if high_performers:
            st.write("The following employees show high job satisfaction and involvement:")
            for emp in high_performers:
                st.markdown(f"**Name:** {emp['Name']}  \n**Job Satisfaction:** {emp['JobSatisfaction']}  \n**Job Involvement:** {emp['JobInvolvement']}")
        else:
            st.info("No high-performing employees found in the sample list.")

    # Feature 4: Side-by-Side Comparisons
    with st.expander("Side-by-Side Employee Comparison"):
        st.subheader("Compare Employee Insights")
        employee_names = [emp['Name'] for emp in sample_employees]
        
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            selected_emp1_name = st.selectbox("Select Employee 1", options=employee_names, key='emp1_select')
        with col_comp2:
            selected_emp2_name = st.selectbox("Select Employee 2", options=employee_names, index=1, key='emp2_select')
        
        emp1_data = next(emp for emp in sample_employees if emp['Name'] == selected_emp1_name)
        emp2_data = next(emp for emp in sample_employees if emp['Name'] == selected_emp2_name)
        
        col_display1, col_display2 = st.columns(2)
        
        def display_employee_details(data):
            st.markdown(f"**Name:** {data['Name']}")
            st.write(f"**Age:** {data['Age']}")
            st.write(f"**Department:** {INV_DEPARTMENT_OPTIONS.get(data['Department'], 'N/A')}")
            st.write(f"**Job Role:** {INV_JOB_ROLE_OPTIONS.get(data['JobRole'], 'N/A')}")
            st.write(f"**Monthly Income:** ${data['MonthlyIncome']:,}")
            st.write(f"**Job Satisfaction:** {data['JobSatisfaction']}")
            st.write(f"**Overtime:** {INV_OVERTIME_OPTIONS.get(data['OverTime'], 'N/A')}")
            st.write(f"**Years at Company:** {data['YearsAtCompany']}")

        with col_display1:
            st.markdown("---")
            display_employee_details(emp1_data)
        with col_display2:
            st.markdown("---")
            display_employee_details(emp2_data)

# Main function
def main():
    st.title("Employee Attrition Prediction")
    
    # Create a form for input
    with st.form("prediction_form"):
        # Personal Information Section
        st.subheader("Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=65, value=30)
            marital_status = st.selectbox("Marital Status", options=list(MARITAL_STATUS_OPTIONS.keys()))
            education_field = st.selectbox("Education Field", options=list(EDUCATION_FIELD_OPTIONS.keys()))
        with col2:
            distance_from_home = st.number_input("Distance from Home (km)", min_value=0, value=5)
            monthly_income = st.number_input("Monthly Income", min_value=1000, value=5000)
            monthly_rate = st.number_input("Monthly Rate", min_value=0, value=10000)
        
        # Job Details Section
        st.subheader("Job Details")
        col3, col4 = st.columns(2)
        
        with col3:
            department = st.selectbox("Department", options=list(DEPARTMENT_OPTIONS.keys()))
            job_role = st.selectbox("Job Role", options=list(JOB_ROLE_OPTIONS.keys()))
            job_level = st.slider("Job Level", min_value=1, max_value=5, value=2)
        with col4:
            overtime = st.selectbox("Overtime", options=list(OVERTIME_OPTIONS.keys()))
            stock_option_level = st.slider("Stock Option Level", min_value=0, max_value=3, value=0)
            num_companies_worked = st.slider("Number of Companies Worked", min_value=0, max_value=9, value=2)
        
        # Satisfaction Section
        st.subheader("Satisfaction & Involvement")
        col5, col6 = st.columns(2)
        
        with col5:
            env_satisfaction = st.slider("Environment Satisfaction", min_value=1, max_value=4, value=2)
            job_satisfaction = st.slider("Job Satisfaction", min_value=1, max_value=4, value=2)
        with col6:
            job_involvement = st.slider("Job Involvement", min_value=1, max_value=4, value=2)
        
        # Experience Section
        st.subheader("Work Experience")
        col7, col8 = st.columns(2)
        
        with col7:
            total_working_years = st.number_input("Total Working Years", min_value=0, value=5)
            years_at_company = st.number_input("Years at Company", min_value=0, value=3)
            years_in_current_role = st.number_input("Years in Current Role", min_value=0, value=2)
        with col8:
            years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, value=1)
            years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0, value=2)
        
        # Submit button
        submit_button = st.form_submit_button("Predict Attrition")
        
    # Make prediction when form is submitted
    if submit_button:
        # Map selections to numeric values
        department_value = DEPARTMENT_OPTIONS[department]
        education_field_value = EDUCATION_FIELD_OPTIONS[education_field]
        job_role_value = JOB_ROLE_OPTIONS[job_role]
        marital_status_value = MARITAL_STATUS_OPTIONS[marital_status]
        overtime_value = OVERTIME_OPTIONS[overtime]
        
        # Create input array
        user_input = [
            age, department_value, distance_from_home, education_field_value,
            env_satisfaction, job_involvement, job_level, job_role_value,
            job_satisfaction, marital_status_value, monthly_income, monthly_rate,
            num_companies_worked, overtime_value, stock_option_level,
            total_working_years, years_at_company, years_in_current_role,
            years_since_last_promotion, years_with_curr_manager
        ]
        
        if model is not None:
            # Make prediction
            input_array = np.array(user_input).reshape(1, -1)
            try:
                prediction = model.predict(input_array)[0]
                probability = model.predict_proba(input_array)[0]
                
                # Display result
                st.markdown("---")
                
                if prediction == 1:
                    st.error("### Prediction: Employee is likely to leave")
                    likelihood = probability[1] * 100
                else:
                    st.success("### Prediction: Employee is likely to stay")
                    likelihood = probability[0] * 100
                
                # Display probability
                st.write(f"Confidence: **{likelihood:.1f}%**")
                
                # Display factors that might affect attrition
                st.subheader("Key Factors to Consider:")
                
                factors = []
                # Use a more robust way to check for 'No' vs 'Yes'
                if overtime == "Yes":
                    factors.append("⚠️ **Overtime:** Might be causing stress and burnout.")
                if job_satisfaction < 3:
                    factors.append("⚠️ **Job Satisfaction:** Current satisfaction is low.")
                if distance_from_home > 10:
                    factors.append("⚠️ **Commute:** Long distance from home.")
                if years_since_last_promotion > 3:
                    factors.append("⚠️ **Promotion:** No recent promotion, which can affect motivation.")
                if env_satisfaction < 3:
                    factors.append("⚠️ **Environment:** Low environment satisfaction.")
                if monthly_income < 3000:
                    factors.append("⚠️ **Monthly Income:** Income is relatively low.")
                
                if factors:
                    for factor in factors:
                        st.write(f"- {factor}")
                else:
                    st.write("No major risk factors identified based on the provided inputs.")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

# Add a navigator to the main page.
def navigator():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Attrition Prediction", "High-Risk Employees", "Job Insights"])

    if page == "Home":
        st.header("Welcome to the Attrition Dashboard")
        st.markdown("Use the navigation on the left to explore the different features of this application. "
                    "You can predict attrition for a specific employee, view a list of high-risk employees, "
                    "or compare different employee profiles.")
        st.markdown("---")
        st.subheader("Final Recommendations:")
        st.write("Improve job satisfaction through engagement initiatives.")
        st.write("Monitor overtime patterns and ensure work-life balance.")
        st.write("Align compensation strategies with industry standards.")
        st.write("Provide growth opportunities to retain top talent.")
        st.write("Use predictive analytics to proactively prevent attrition.")
        st.write("Promoting high-performance employees helps prevent attrition and improves job satisfaction.")

    elif page == "Attrition Prediction":
        main()
    elif page == "High-Risk Employees":
        with st.expander("High-Risk Employee List", expanded=True):
            st.subheader("Employees with High Attrition Risk")
            high_risk_employees = []
            if model is not None:
                for emp in sample_employees:
                    user_input = get_employee_features(emp)
                    input_array = np.array(user_input).reshape(1, -1)
                    try:
                        probability = model.predict_proba(input_array)[0]
                        attrition_prob = probability[1]
                        if attrition_prob > 0.5:
                            emp_copy = emp.copy()
                            emp_copy['Attrition Probability'] = f"{attrition_prob:.2f}"
                            high_risk_employees.append(emp_copy)
                    except Exception as e:
                        st.warning(f"Could not predict for {emp['Name']}: {str(e)}")
                
                if high_risk_employees:
                    df = pd.DataFrame(high_risk_employees)
                    st.table(df[['Name', 'Attrition Probability', 'OverTime', 'JobSatisfaction', 'MonthlyIncome']])
                else:
                    st.info("No employees identified as high-risk in the sample list.")
    elif page == "Job Insights":
        with st.expander("Job Satisfaction & Performance Insights", expanded=True):
            st.subheader("High Job Satisfaction & Performance")
            high_performers = [emp for emp in sample_employees if emp['JobSatisfaction'] >= 3 and emp['JobInvolvement'] >= 3]
            if high_performers:
                st.write("The following employees show high job satisfaction and involvement:")
                for emp in high_performers:
                    st.markdown(f"**Name:** {emp['Name']}  \n**Job Satisfaction:** {emp['JobSatisfaction']}  \n**Job Involvement:** {emp['JobInvolvement']}")
            else:
                st.info("No high-performing employees found in the sample list.")
        
        with st.expander("Side-by-Side Employee Comparison", expanded=True):
            st.subheader("Compare Employee Insights")
            employee_names = [emp['Name'] for emp in sample_employees]
            
            col_comp1, col_comp2 = st.columns(2)
            
            with col_comp1:
                selected_emp1_name = st.selectbox("Select Employee 1", options=employee_names, key='emp1_select')
            with col_comp2:
                selected_emp2_name = st.selectbox("Select Employee 2", options=employee_names, index=1, key='emp2_select')
            
            emp1_data = next(emp for emp in sample_employees if emp['Name'] == selected_emp1_name)
            emp2_data = next(emp for emp in sample_employees if emp['Name'] == selected_emp2_name)
            
            col_display1, col_display2 = st.columns(2)
            
            def display_employee_details(data):
                st.markdown(f"**Name:** {data['Name']}")
                st.write(f"**Age:** {data['Age']}")
                st.write(f"**Department:** {INV_DEPARTMENT_OPTIONS.get(data['Department'], 'N/A')}")
                st.write(f"**Job Role:** {INV_JOB_ROLE_OPTIONS.get(data['JobRole'], 'N/A')}")
                st.write(f"**Monthly Income:** ${data['MonthlyIncome']:,}")
                st.write(f"**Job Satisfaction:** {data['JobSatisfaction']}")
                st.write(f"**Overtime:** {INV_OVERTIME_OPTIONS.get(data['OverTime'], 'N/A')}")
                st.write(f"**Years at Company:** {data['YearsAtCompany']}")

            with col_display1:
                st.markdown("---")
                display_employee_details(emp1_data)
            with col_display2:
                st.markdown("---")
                display_employee_details(emp2_data)

if __name__ == "__main__":
    # Start the app with the navigator
    navigator()