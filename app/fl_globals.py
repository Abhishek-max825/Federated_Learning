from app.fl.aggregator import FLAggregator
from app.fl.ecg_aggregator import FedBNAggregator

# Heart Disease FL Aggregator (FedAvg for tabular data)
heart_aggregator = FLAggregator()

# ECG Arrhythmia FL Aggregator (FedBN for CNN with BatchNorm)
ecg_aggregator = FedBNAggregator()

# Backward compatibility - default to heart aggregator
aggregator = heart_aggregator
