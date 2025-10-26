import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, RegressionPreset
from evidently.metrics import * 

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import *

st.markdown("# Inference dataset and data drift report")
st.sidebar.markdown("# Browse a test dataset for inference")

browse_test_file = st.container()
show_imported_dataset = st.container()
evidently = st.container()
live_streaming = st.container()

def get_raw_data():
    df_train = pd.read_csv(r'C:\Users\david\Programming\MLOps\MLSecOps_E1DKHM_Projektmunka\data\train.csv')
    df_test = pd.read_csv(r'C:\Users\david\Programming\MLOps\MLSecOps_E1DKHM_Projektmunka\data\test.csv')
    experiment_name = '1'
    
    return df_train, df_test

df_train, df_test = get_raw_data()

with browse_test_file:
    uploaded_file = st.file_uploader("Please browse a test dataset for inference")
    
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        imported_test_df = pd.read_csv(uploaded_file)
        #st.write(imported_test_df.head(20))


if uploaded_file is not None:
    with show_imported_dataset:
        st.header("It is the head of the dataset what you imported.")

        st.write(imported_test_df.head(20))
        
    with evidently:
        st.header("Data drift report")

        reference = df_train.copy()
        current = imported_test_df.copy()

        report = Report(metrics=[
        DataDriftPreset(), 
            ])

        report.run(reference_data=reference, current_data=current)

        report.save_html('report.html')

        HtmlFile = open("report.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        #print(source_code)
        components.html(source_code, height=2000)
       
    # Live streaming
    import time
    #!/usr/bin/env python
    import psutil
    
    index = 0
    cpu_load = []
    with live_streaming:
        st.header("Streaming plot 'CPU load'")

        # gives a single float value
        cpu_load.append(psutil.cpu_percent())
        # gives an object with many fields
        chart = st.empty()
        chart.line_chart(pd.Series(cpu_load))

        while True:
            cpu_load.append(psutil.cpu_percent())
            if len(pd.Series(cpu_load)) > 20:
                cpu_list = pd.Series(cpu_load).iloc[index-20:index]
            else:
                cpu_list = pd.Series(cpu_load)

            # Clear all those elements:
            chart.empty()
            chart.line_chart(cpu_list)
            
            index += 1
            time.sleep(1)
 