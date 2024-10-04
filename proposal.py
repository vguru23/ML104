import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

# Contribution data
data = {
    "Name": ["Saloni Jain", "Sneha Pal", "Sooriya Senthilkumar", "Aneesh Sabarad", "Vibha Guru"],
    "Contribution": [
        "Worked on Proposal",
        "Worked on Proposal",
        "Created Powerpoint and Gantt Chart",
        "Created Video",
        "Worked on proposal & Streamlit page"
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Centered title using HTML
st.markdown(
    "<h1 style='text-align: center;'>Group 104 - ML Proposal</h1>", 
    unsafe_allow_html=True
)

# Sections with dropdowns
with st.expander("Introduction and Literature Review"):
    st.write("""
    People's physical capabilities change over time, which can be challenging for those with disabilities or chronic conditions. 
    Therefore, a dynamic smart bathroom environment can be beneficial for health monitoring and diagnosing[1]. 
    In addition, monitoring physical activity with sensors and force readings can help create tailored treatments quickly and in real time[2]. 
    A smart bathroom environment can also be used for documenting physical data and identifying behavioral changes[3]. 
    The goal of our research is to develop a machine learning model that classifies movement as a person sits on the toilet into three classes: 
    sitting, offboarding, and onboarding to identify movement patterns in adults with physical impairments.
    """)

with st.expander("Dataset Description"):
    st.write("""
    Our dataset consists of force sensor readings from 40 people with 20 trials each. Some key features include:
    - Timestamps (Unix format): These represent the exact time when the sensor data was captured.
    - Load sensor values: These are readings from several sensors scattered around the toilet seat. 
      These numbers will be useful in determining the Center of Pressure (COP).
    """)
    st.write("Dataset Link: https://tinyurl.com/toiletseatdata")

with st.expander("Problem and Motivation"):
    st.write("""
    The purpose of this research is to detect unexpected behaviors such as abnormal sitting or rising patterns in participants with physical disabilities. 
    This technology can be integrated into alarm systems that alert family members or caretakers in the event of a fall or prolonged sitting. 
    In addition, this technology can help monitor changes in movement patterns of those with disabilities to monitor long term health.
    """)

with st.expander("Data Preprocessing Methods"):
    st.subheader("Data Cleaning and Selection")
    st.write("""
    We will remove all columns except the last 4 toilet seat data and time columns. 
    We will convert the Unix timestamp to the number of seconds passed since data collection began.
    """)

    st.subheader("Noise Reduction")
    st.write("""
    To reduce noise from the raw sensor data, we'll apply a low-pass filter. 
    This will smooth out rapid fluctuations while preserving the overall trends in force measurements.
    """)

    st.subheader("Feature Extraction")
    st.write("""
    We'll calculate the Center of Pressure (COP) using the four load sensor values to calculate a single metric 
    representing the overall pressure distribution.
    """)

with st.expander("ML Models/Algorithms"):
    st.subheader("Random Forest")
    st.write("""
    We'll use scikit-learn's RandomForestClassifier to differentiate between the three states (onboarding, sitting, off-boarding). 
    Random forests can capture non-linear relationships in data.
    """)

    st.subheader("Support Vector Machine (SVM)")
    st.write("""
    We'll implement an SVM using scikit-learn's SVC class which is accurate at finding optimal segments between classes in high-dimensional spaces.
    """)

    st.subheader("Logistic Regression")
    st.write("""
    We will use scikit-learn's LogisticRegression as a baseline model. 
    Logistic regression can serve as a good comparison point for more complex models.
    """)

with st.expander("Quantitative Metrics"):
    st.subheader("Precision")
    st.write("Ratio of correctly predicted positive instances to total predicted positive instances.")
    st.write("(true positives) / (false positives + true positives)")

    st.subheader("Accuracy")
    st.write("Ratio of correctly predicted instances to total number of instances")
    st.write("(true positives + true negatives) / all instances")

    st.subheader("F1 Score")
    st.write("Mean of precision and recall")
    st.write("(F1 Score=2×(precision*recall / precision + recall)")

    st.subheader("Recall")
    st.write("Ratio of correctly predicted positive instances to all actual positive instances")
    st.write("(true positives) / (false negatives + true positives)")

with st.expander("Project Goals and Expected Results"):
    st.write("""
    The project goal would be to have overall high precision and accuracy which therefore leads to a high F1 score. 
    A high recall is also important because of the need to minimize the number of false negatives in a medical setting. 
    One ethical consideration is if the model is wrong and does not find when a participant is struggling with their movement patterns. 
    Conversely, the model could wrongly identify patients with abnormal movement patterns. 
    For expected results, the model should be able to classify a time range in which a participant is getting on/off the toilet seat.
    """)

# Gantt chart section with link
with st.expander("Gantt Chart"):
   st.markdown("[View Gantt Chart in Google Sheets](https://docs.google.com/spreadsheets/d/1pWEyieNCmKAQnlG2C3LrY10Mgpy52Tdx/edit?usp=sharing&ouid=108969903742919067214&rtpof=true&sd=true)")

# Contribution table
with st.expander("Contribution Table"):
   st.table(df)

# Proposal Video
with st.expander("Proposal Video"):
   st.markdown("[View Video on Youtube](https://youtu.be/OhnM3QEBTxs)")

# Citations
with st.expander("Citations"):
    st.write("""
    [1] B. Jones et al., "Smart Bathroom: Developing a Smart Environment to Study Bathroom Transfers," 2017, https://www.resna.org/sites/default/files/conference/2017/pdf_versions/outcomes/Jones.pdf

    [2] J. Pohl et al., "Accuracy of gait and posture classification using movement sensors in individuals with mobility impairment after stroke," Frontiers in Physiology, vol. 13, Sep. 2022, doi: https://doi.org/10.3389/fphys.2022.933987.

    [3] S. Hermsen, V. Verbiest, Marije Buijs, and E. Wentink, "Perceived Use Cases, Barriers, and Requirements for a Smart Health-Tracking Toilet Seat: Qualitative Focus Group Study," JMIR human factors, vol. 10, pp. e44850–e44850, Aug. 2023, doi: https://doi.org/10.2196/44850.
    """)