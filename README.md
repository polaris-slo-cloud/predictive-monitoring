# High-level predictive monitoring

This repository contains the code to generate LSTM and transformer models for high-level monitoring prediction or to execute an already trained one.
The data folder contains pre-filtered and pre-processed data from [Google Cluster Data - 2011-2](https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md).

The folder [models](https://github.com/polaris-slo-cloud/predictive-monitoring/tree/master/models/lstm_batch72_neurons50_epochs400_do0) contains the pre-trained LSTMs. To test it, you can run the test: `python gcd_test_model.py 6318371744`; the second argument represents the ID of the job to consider. The Jupyter notebook [`test_gcd-model_predictions.ipynb`](https://github.com/polaris-slo-cloud/predictive-monitoring/tree/master/lstm_approach/test_gcd-model_predictions.ipynb) offers the possibility to explore different ways to test new data.

The folder [`lstm_approach`](https://github.com/polaris-slo-cloud/predictive-monitoring/tree/master/lstm_approach) contains the code to run the LSTM model. To re-train the LSTM, it is possible to run the script `gcd_single-job_multivariate_prediction.py [epochs neurons batch_size] --exp-name [exp]`. To reproduce the exact same model, the code is:`gcd_single-job_multivariate_prediction.py 400 50 72 --exp_name exp_01`.

The folder [`transformer_approach`](https://github.com/polaris-slo-cloud/predictive-monitoring/tree/master/transformer_approach) contains all the required components related with the transformer model, as well as a detailed readme file. There, one can find the model used for the resource prediction inside the [`model` folder](https://github.com/polaris-slo-cloud/predictive-monitoring/tree/master/transformer_approach/models). Also, a simplified python code called [`example.py`](https://github.com/polaris-slo-cloud/predictive-monitoring/tree/master/transformer_approach/example.py) is provided in order to test and learn how to use the model.
