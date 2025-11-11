"""
Parkinson's Disease Prediction Web Application
Streamlit UI for Real-time Prediction with Severity Assessment
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Parkinson's Disease Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
    }
    .positive {
        background-color: #ffebee;
        border: 3px solid #ef5350;
        color: #c62828;
    }
    .negative {
        background-color: #e8f5e9;
        border: 3px solid #66bb6a;
        color: #2e7d32;
    }
    .severity-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        background-color: #00008B;
        border: 2px solid #FFFFFF;
    }
    .info-box {
        padding: 1rem;
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load models and preprocessing objects
@st.cache_resource
def load_models():
    try:
        model_path = 'models/best_model.pkl'
        scaler_path = 'models/scaler.pkl'
        selector_path = 'models/feature_selector.pkl'
        metadata_path = 'models/model_metadata.pkl'
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        selector = joblib.load(selector_path)
        metadata = joblib.load(metadata_path)
        
        return model, scaler, selector, metadata
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please ensure model files are in the 'models/' directory")
        return None, None, None, None

# Feature engineering function
def engineer_features(df):
    """Apply same feature engineering as training"""
    
    # Age features
    df['Age_Squared'] = df['Age'] ** 2
    df['Age_Cubed'] = df['Age'] ** 3
    df['Age_Log'] = np.log1p(df['Age'])
    df['Age_Group_Young'] = (df['Age'] < 60).astype(int)
    df['Age_Group_Middle'] = ((df['Age'] >= 60) & (df['Age'] < 75)).astype(int)
    df['Age_Group_Senior'] = (df['Age'] >= 75).astype(int)
    
    # BMI features
    df['BMI_Squared'] = df['BMI'] ** 2
    df['BMI_Underweight'] = (df['BMI'] < 18.5).astype(int)
    df['BMI_Normal'] = ((df['BMI'] >= 18.5) & (df['BMI'] < 25)).astype(int)
    df['BMI_Overweight'] = ((df['BMI'] >= 25) & (df['BMI'] < 30)).astype(int)
    df['BMI_Obese'] = (df['BMI'] >= 30).astype(int)
    
    # Symptom aggregation
    symptom_cols = ['Tremor', 'Rigidity', 'Bradykinesia', 'PosturalInstability',
                    'SpeechProblems', 'SleepDisorders', 'Constipation']
    df['Total_Symptoms'] = df[symptom_cols].sum(axis=1)
    df['Symptom_Percentage'] = df['Total_Symptoms'] / len(symptom_cols)
    df['Motor_Symptoms'] = df[['Tremor', 'Rigidity', 'Bradykinesia', 'PosturalInstability']].sum(axis=1)
    df['NonMotor_Symptoms'] = df[['SpeechProblems', 'SleepDisorders', 'Constipation']].sum(axis=1)
    df['Has_All_Motor'] = (df['Motor_Symptoms'] == 4).astype(int)
    df['Has_No_Symptoms'] = (df['Total_Symptoms'] == 0).astype(int)
    
    # Medical history
    medical_cols = ['FamilyHistoryParkinsons', 'TraumaticBrainInjury', 
                    'Hypertension', 'Diabetes', 'Depression', 'Stroke']
    df['Total_Medical_Conditions'] = df[medical_cols].sum(axis=1)
    df['Medical_Risk_Score'] = df['Total_Medical_Conditions'] / len(medical_cols)
    df['Has_Multiple_Conditions'] = (df['Total_Medical_Conditions'] >= 2).astype(int)
    df['Neurological_History'] = df[['FamilyHistoryParkinsons', 'TraumaticBrainInjury', 'Stroke']].sum(axis=1)
    
    # Lifestyle features
    df['Lifestyle_Score'] = (df['PhysicalActivity'] + df['DietQuality'] + df['SleepQuality']) / 3
    df['Lifestyle_Good'] = (df['Lifestyle_Score'] > 6).astype(int)
    df['Lifestyle_Poor'] = (df['Lifestyle_Score'] < 4).astype(int)
    df['Risk_Factors'] = df['Smoking'] + (df['AlcoholConsumption'] / 20)
    df['High_Risk_Lifestyle'] = ((df['Smoking'] == 1) & (df['AlcoholConsumption'] > 10)).astype(int)
    df['Healthy_Lifestyle'] = ((df['PhysicalActivity'] > 5) & (df['DietQuality'] > 6) & 
                               (df['Smoking'] == 0)).astype(int)
    
    # Cardiovascular features
    df['BP_Product'] = df['SystolicBP'] * df['DiastolicBP']
    df['BP_Sum'] = df['SystolicBP'] + df['DiastolicBP']
    df['BP_Ratio'] = df['SystolicBP'] / (df['DiastolicBP'] + 1)
    df['Hypertension_Risk'] = ((df['SystolicBP'] >= 140) | (df['DiastolicBP'] >= 90)).astype(int)
    df['Pulse_Pressure'] = df['SystolicBP'] - df['DiastolicBP']
    
    # Cholesterol features
    df['Cholesterol_Ratio'] = df['CholesterolLDL'] / (df['CholesterolHDL'] + 1)
    df['Cholesterol_Risk'] = (df['CholesterolTotal'] > 240).astype(int)
    df['Atherogenic_Index'] = (df['CholesterolTotal'] - df['CholesterolHDL']) / (df['CholesterolHDL'] + 1)
    df['HDL_LDL_Diff'] = df['CholesterolLDL'] - df['CholesterolHDL']
    df['High_Triglycerides'] = (df['CholesterolTriglycerides'] > 150).astype(int)
    
    # Cognitive features
    df['Cognitive_Functional_Product'] = df['MoCA'] * df['FunctionalAssessment']
    df['Cognitive_Functional_Sum'] = df['MoCA'] + df['FunctionalAssessment']
    df['Cognitive_Functional_Ratio'] = df['MoCA'] / (df['FunctionalAssessment'] + 1)
    df['Cognitive_Impairment'] = (df['MoCA'] < 26).astype(int)
    df['Functional_Impairment'] = (df['FunctionalAssessment'] < 5).astype(int)
    df['Severe_Cognitive_Decline'] = (df['MoCA'] < 18).astype(int)
    
    # Interaction features
    df['Age_BMI_Interaction'] = df['Age'] * df['BMI']
    df['Age_Symptoms_Interaction'] = df['Age'] * df['Total_Symptoms']
    df['Age_UPDRS_Interaction'] = df['Age'] * df['UPDRS']
    df['BMI_UPDRS_Interaction'] = df['BMI'] * df['UPDRS']
    df['Symptoms_Medical_Interaction'] = df['Total_Symptoms'] * df['Total_Medical_Conditions']
    
    # Gender-specific features
    df['Male_Age'] = df['Gender'].apply(lambda x: 0 if x == 1 else 1) * df['Age']
    df['Female_Age'] = df['Gender'] * df['Age']
    df['Male_High_Risk'] = ((df['Gender'] == 0) & (df['Age'] > 65)).astype(int)
    df['Female_High_Risk'] = ((df['Gender'] == 1) & (df['Age'] > 70)).astype(int)
    
    # Severity indicators
    df['UPDRS_Squared'] = df['UPDRS'] ** 2
    df['UPDRS_Log'] = np.log1p(df['UPDRS'])
    df['Severe_UPDRS'] = (df['UPDRS'] > 95).astype(int)
    df['Mild_UPDRS'] = (df['UPDRS'] < 32).astype(int)
    df['UPDRS_Category'] = pd.cut(df['UPDRS'], bins=[0, 32, 58, 95, 200], labels=[0, 1, 2, 3]).astype(int)
    
    # Composite risk scores
    df['Overall_Health_Score'] = (
        df['Lifestyle_Score'] * 0.3 + 
        (10 - df['Medical_Risk_Score'] * 10) * 0.3 + 
        (df['MoCA'] / 3) * 0.2 +
        (df['FunctionalAssessment']) * 0.2
    )
    df['Parkinson_Risk_Score'] = (
        df['Age'] / 100 * 0.2 +
        df['FamilyHistoryParkinsons'] * 0.15 +
        df['Total_Symptoms'] / 7 * 0.25 +
        df['UPDRS'] / 200 * 0.25 +
        (1 - df['MoCA'] / 30) * 0.15
    )
    
    return df

def get_severity_level(updrs_score):
    """Determine severity level from UPDRS score"""
    if updrs_score <= 32:
        return "Minimal", "#4caf50"
    elif updrs_score <= 58:
        return "Mild", "#ffeb3b"
    elif updrs_score <= 95:
        return "Moderate", "#ff9800"
    else:
        return "Severe", "#f44336"

# Main application
def main():
    # Header
    st.markdown('<p class="main-header">🧠 Parkinson\'s Disease Prediction System</p>', unsafe_allow_html=True)
    
    # Load models
    model, scaler, selector, metadata = load_models()
    
    if model is None:
        st.error("Failed to load models. Please check model files.")
        return
    
    # Sidebar - Model info
    with st.sidebar:
        st.header("📊 Model Information")
        st.info(f"""
        **Model:** {metadata['best_model_name']}
        
        **Performance:**
        - Accuracy: {metadata['test_accuracy']*100:.2f}%
        - F1-Score: {metadata['test_f1']:.4f}
        - ROC-AUC: {metadata['test_roc_auc']:.4f}
        
        **Training Date:** {metadata['training_date']}
        """)
        
        st.header("ℹ️ About")
        st.write("""
        This AI-powered system predicts Parkinson's Disease risk 
        and estimates severity based on clinical and demographic data.
        
        **Accuracy:** 90%+
        
        **Features Used:** 40+ engineered features
        """)
    
    # Main content
    st.markdown("### 📝 Enter Patient Information")
    
    # Create tabs for input
    tab1, tab2, tab3, tab4 = st.tabs(["👤 Demographics", "🏥 Medical History", "💊 Clinical Measurements", "🧪 Assessments & Symptoms"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=50, max_value=90, value=65, help="Patient's age in years")
            gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
            
        with col2:
            bmi = st.number_input("BMI", min_value=15.0, max_value=40.0, value=25.0, step=0.1)
            smoking = st.selectbox("Smoking", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col3:
            alcohol = st.slider("Alcohol Consumption (units/week)", min_value=0, max_value=20, value=5)
            physical_activity = st.slider("Physical Activity (hours/week)", min_value=0, max_value=10, value=5)
            diet_quality = st.slider("Diet Quality (0-10)", min_value=0, max_value=10, value=5)
            sleep_quality = st.slider("Sleep Quality (4-10)", min_value=4, max_value=10, value=7)
    
    with tab2:
      col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Family & Neurological History:**")
        family_history = st.checkbox("Family History of Parkinson's")
        traumatic_brain_injury = st.checkbox("Traumatic Brain Injury")
        stroke = st.checkbox("History of Stroke")
    
    with col2:
        st.markdown("**Chronic Conditions:**")
        hypertension = st.checkbox("Hypertension")
        diabetes = st.checkbox("Diabetes")
        depression = st.checkbox("Depression")

    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Blood Pressure:**")
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=90, max_value=180, value=120)
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=60, max_value=120, value=80)
            
            st.markdown("**Cholesterol Levels:**")
            cholesterol_total = st.number_input("Total Cholesterol (mg/dL)", min_value=150, max_value=300, value=200)
            cholesterol_ldl = st.number_input("LDL Cholesterol (mg/dL)", min_value=50, max_value=200, value=100)
        
        with col2:
            st.markdown("**Additional Cholesterol:**")
            cholesterol_hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=20, max_value=100, value=50)
            cholesterol_triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=50, max_value=400, value=150)
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Cognitive & Functional Assessments:**")
            updrs = st.slider("UPDRS Score (0-199)", min_value=0, max_value=199, value=50, 
                            help="Unified Parkinson's Disease Rating Scale")
            moca = st.slider("MoCA Score (0-30)", min_value=0, max_value=30, value=25,
                           help="Montreal Cognitive Assessment")
            functional_assessment = st.slider("Functional Assessment (0-10)", min_value=0, max_value=10, value=7)
        
        with col2:
            st.markdown("**Symptoms:**")
            tremor = st.checkbox("Tremor")
            rigidity = st.checkbox("Rigidity")
            bradykinesia = st.checkbox("Bradykinesia (slowness of movement)")
            postural_instability = st.checkbox("Postural Instability")
            speech_problems = st.checkbox("Speech Problems")
            sleep_disorders = st.checkbox("Sleep Disorders")
            constipation = st.checkbox("Constipation")
    
    # Predict button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("🔍 PREDICT", use_container_width=True, type="primary")
    
    if predict_button:
        # Create input dataframe
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'BMI': [bmi],
            'Smoking': [smoking],
            'AlcoholConsumption': [alcohol],
            'PhysicalActivity': [physical_activity],
            'DietQuality': [diet_quality],
            'SleepQuality': [sleep_quality],
            'FamilyHistoryParkinsons': [int(family_history)],
            'TraumaticBrainInjury': [int(traumatic_brain_injury)],
            'Hypertension': [int(hypertension)],
            'Diabetes': [int(diabetes)],
            'Depression': [int(depression)],
            'Stroke': [int(stroke)],
            'SystolicBP': [systolic_bp],
            'DiastolicBP': [diastolic_bp],
            'CholesterolTotal': [cholesterol_total],
            'CholesterolLDL': [cholesterol_ldl],
            'CholesterolHDL': [cholesterol_hdl],
            'CholesterolTriglycerides': [cholesterol_triglycerides],
            'UPDRS': [updrs],
            'MoCA': [moca],
            'FunctionalAssessment': [functional_assessment],
            'Tremor': [int(tremor)],
            'Rigidity': [int(rigidity)],
            'Bradykinesia': [int(bradykinesia)],
            'PosturalInstability': [int(postural_instability)],
            'SpeechProblems': [int(speech_problems)],
            'SleepDisorders': [int(sleep_disorders)],
            'Constipation': [int(constipation)]
        })
        
        try:
            for col in ["UPDRS",'EducationLevel', 'Ethnicity']:
                if col not in input_data.columns:
                    input_data[col] = 0
            # Apply feature engineering
            input_data_engineered = engineer_features(input_data)

            expected_features = selector.feature_names_in_
            input_data_aligned = input_data_engineered.reindex(columns=expected_features, fill_value=0)

            # Apply selector and scaler
            input_selected = selector.transform(input_data_aligned)
            # ✅ Step 1: Apply feature selector first (ensure correct 40 features)
            # input_selected = selector.transform(input_data_engineered)

            # ✅ Step 2: Scale only the selected features
            input_scaled = scaler.transform(input_selected)

            # ✅ Step 3: Predict
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Prediction Results")
            
            # Main prediction
            if prediction == 1:
                confidence = prediction_proba[1] * 100
                st.markdown(f"""
                <div class="prediction-box positive">
                    ⚠️ PARKINSON'S DISEASE DETECTED<br>
                    Confidence: {confidence:.1f}%
                </div>
                """, unsafe_allow_html=True)
                
                # Severity assessment
                severity_level, severity_color = get_severity_level(updrs)
                st.markdown(f"""
                <div class="severity-box">
                    <h3 style="color: {severity_color};">Severity Level: {severity_level}</h3>
                    <p><strong>UPDRS Score:</strong> {updrs} / 199</p>
                    <p><strong>Severity Classification:</strong></p>
                    <ul>
                        <li>Minimal: 0-32</li>
                        <li>Mild: 33-58</li>
                        <li>Moderate: 59-95</li>
                        <li>Severe: 96+</li>
                    </ul>
                    <p><strong>Current Status:</strong> {severity_level.upper()}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Recommendations
                st.markdown("### 🏥 Recommendations")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info("""
                    **Immediate Actions:**
                    - Consult a neurologist immediately
                    - Schedule comprehensive neurological examination
                    - Consider MRI/CT imaging if not done recently
                    - Start symptom tracking diary
                    """)
                
                with col2:
                    st.warning("""
                    **Lifestyle Modifications:**
                    - Increase physical activity gradually
                    - Follow Mediterranean diet
                    - Maintain regular sleep schedule
                    - Join Parkinson's support groups
                    """)
                
                # Risk factors
                st.markdown("### ⚠️ Key Risk Factors Identified")
                risk_factors = []
                
                if age > 70:
                    risk_factors.append(f"Advanced age ({age} years)")
                if family_history:
                    risk_factors.append("Family history of Parkinson's")
                if updrs > 95:
                    risk_factors.append(f"High UPDRS score ({updrs})")
                if moca < 26:
                    risk_factors.append(f"Cognitive impairment (MoCA: {moca})")
                if tremor and rigidity and bradykinesia:
                    risk_factors.append("Multiple motor symptoms present")
                
                if risk_factors:
                    for rf in risk_factors:
                        st.markdown(f"- 🔴 {rf}")
                else:
                    st.success("No major risk factors identified")
                
            else:
                confidence = prediction_proba[0] * 100
                st.markdown(f"""
                <div class="prediction-box negative">
                    ✅ NO PARKINSON'S DISEASE DETECTED<br>
                    Confidence: {confidence:.1f}%
                </div>
                """, unsafe_allow_html=True)
                
                # Preventive measures
                st.markdown("### 💚 Preventive Measures")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success("""
                    **Maintain Healthy Lifestyle:**
                    - Regular exercise (150+ min/week)
                    - Balanced diet rich in antioxidants
                    - 7-9 hours quality sleep
                    - Stress management techniques
                    """)
                
                with col2:
                    st.info("""
                    **Regular Monitoring:**
                    - Annual health check-ups
                    - Monitor any new symptoms
                    - Track family health history
                    - Stay informed about Parkinson's research
                    """)
            
            # Detailed metrics
            st.markdown("---")
            st.markdown("### 📈 Detailed Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Age Risk", f"{age} years", 
                         "High" if age > 70 else "Moderate" if age > 60 else "Low")
            
            with col2:
                symptom_count = sum([tremor, rigidity, bradykinesia, postural_instability, 
                                   speech_problems, sleep_disorders, constipation])
                st.metric("Symptom Count", f"{symptom_count}/7",
                         "High" if symptom_count >= 4 else "Moderate" if symptom_count >= 2 else "Low")
            
            with col3:
                st.metric("MoCA Score", f"{moca}/30",
                         "Normal" if moca >= 26 else "Impaired")
            
            with col4:
                st.metric("UPDRS Score", f"{updrs}/199",
                         severity_level)
            
            # Probability breakdown
            st.markdown("### 🎯 Prediction Confidence")
            col1, col2 = st.columns(2)
            
            with col1:
                st.progress(prediction_proba[0])
                st.caption(f"No Parkinson's: {prediction_proba[0]*100:.2f}%")
            
            with col2:
                st.progress(prediction_proba[1])
                st.caption(f"Parkinson's: {prediction_proba[1]*100:.2f}%")
            
            # Disclaimer
            st.markdown("---")
            # st.markdown("""
            # <div class="info-box">
            # <strong>⚠️ Medical Disclaimer:</strong><br>
            # This prediction system is for informational purposes only and should not replace 
            # professional medical advice, diagnosis, or treatment. Always consult with a qualified 
            # healthcare provider for proper diagnosis and treatment recommendations.
            # </div>
            # """, unsafe_allow_html=True)
            
            # Download report
            st.markdown("---")
            report_data = {
                'Prediction': ['Parkinson\'s Disease' if prediction == 1 else 'No Parkinson\'s Disease'],
                'Confidence': [f"{max(prediction_proba)*100:.2f}%"],
                'UPDRS Score': [updrs],
                'Severity': [severity_level if prediction == 1 else 'N/A'],
                'MoCA Score': [moca],
                'Age': [age],
                'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            }
            report_df = pd.DataFrame(report_data)
            
            csv = report_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Prediction Report",
                data=csv,
                file_name=f"parkinsons_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("Please check all inputs and try again")

if __name__ == "__main__":
    main()
