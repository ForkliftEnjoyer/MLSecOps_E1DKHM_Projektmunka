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

st.markdown("# Train Dataset ")
st.sidebar.markdown("# Train Dataset ")

def get_raw_data():
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')
    experiment_name = '1'
    
    return df_train, df_test

header = st.container()
dataset = st.container()
plot_area_code = st.container()

with header:
    st.title('Monitoring some elements')
    st.text("You can see some examples of Streamlit'possibilities")

with dataset:
    st.header("Churn dataset import")
    st.text("You can see here a sample from the train dataset")
    
    df_train, df_test = get_raw_data()
    st.write(df_train.head(20))


with plot_area_code:
    st.header("Plot Area code value counts")
    total_day_minutes = df_train["Survived"].value_counts().head(50)
    st.bar_chart(total_day_minutes)

# streamlit run monitor_with_streamlit_train_data.py