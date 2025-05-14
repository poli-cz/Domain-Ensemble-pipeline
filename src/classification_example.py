# Import necessary modules
import sys

from core.validator import load_saved_split, load_train_split, load_random_sample
from pipeline import DomainClassifier


STAGE = 3  # 1 / 2 / 3

x_test, y_test = load_random_sample(STAGE, folder="./data/")

# Initialize classifier
dc_phish = DomainClassifier(data_sample=x_test, label="phishing")
dc_phish.determine_stage(x_test)

dc_malw = DomainClassifier(data_sample=x_test, label="malware")
dc_malw.determine_stage(x_test)


# Classify domains one by one
for domain, expected_label in zip(x_test, y_test):
    # Classify the domain
    phishing_prediction = dc_phish.classify(domain)

    malware_prediction = dc_malw.classify(domain)

    # Print the result
    print(
        f"Expected: {expected_label}, Malware: {malware_prediction['meta_proba']}, Phishing: {phishing_prediction['meta_proba']}"
    )
    input(
        "Press Enter to continue..."
    )  # Wait for user input before proceeding to the next domain
