# educational-risk-measure
### Interactive form for measuring level of academic performance risk built on Streamlit and driven by a combined ML model created with usage of scikit-learn, xgboost, lightgbm and catboost on Python.

Model is trained on a Kaggle dataset on behavioral psychology, welness, career development goals and objective academic performance on more than 1000 students (link to it: https://www.kaggle.com/datasets/nawazkhan3251/student-performance-and-behavioral-analytics-dataset). Overall accuracy on the test dataset is somewhere around 97.5-98 percent. Interactive interface is implemented on a local host using Streamlit tools. Results of model training are located in 'model' folder.

### How to run an application

1) Download all files from 'model' folder or 'EnsembleModel' file if you want to train modle by yourself or modify it somehow (In this case, you also need to upload a .csv file and locate it in the same folder as model)
2) Download 'StreamlitRiskPoll' and locate this file in the same folder as your .pkl files or a .py model.
3) Run a program in your preferred coding environment
4) Run an application on a local host. Instructions on how to do it: https://docs.streamlit.io/develop/concepts/architecture/run-your-app
