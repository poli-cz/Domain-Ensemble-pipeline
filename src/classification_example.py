# Import necessary modules
import sys

from core.validator import load_saved_split, load_train_split
from pipeline import DomainClassifier


MALICIOUS_LABEL = "phishing"  # phishing / malware
STAGE = 3  # 1 / 2 / 3
VERIFICATION = True  # True / False, use verification dataset of validation dataset


x_test, y_test = load_saved_split(
    STAGE, MALICIOUS_LABEL, folder="./data/", verification=VERIFICATION
)

# Initialize classifier
DomainClassifier = DomainClassifier(data_sample=x_test, label=MALICIOUS_LABEL)
DomainClassifier.determine_stage(x_test)


# Classify domains one by one
for domain, expected_label in zip(x_test, y_test):
    # Classify the domain
    predicted_label = DomainClassifier.classify(domain)

    # Print the result
    print(f"Expected: {expected_label}, Predicted: {predicted_label}")
    input(
        "Press Enter to continue..."
    )  # Wait for user input before proceeding to the next domain
