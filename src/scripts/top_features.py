import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Features from each dataset
features = [
    "SubjectCommonNameIp",
    "Is_extended_validated",
    "Is_organization_validated",
    "Is_domian_validated",
    "SubjectHasOrganization",
    "IssuerHasOrganization",
    "SubjectHasCompany",
    "IssuerHasCompany",
    "SubjectHasState",
    "IssuerHasState",
    "SubjectHasLocation",
    "IssuerHasLocation",
    "Subject_onlyCN",
    "Subject_is_com",
    "Issuer_is_com",
    "HasSubjectCommonName",
    "HasIssuerCommonName",
    "Subject_eq_Issuer",
    "SubjectElements",
    "IssuerElements",
    "SubjectLength",
    "IssuerLength",
    "ExtensionNumber",
    "Selfsigned",
    "Is_free",
    "DaysValidity",
    "Ranking_C",
    "SubjectCommonName",
    "Euclidian_Subject_Subjects",
    "Euclidian_Subject_English",
    "Euclidian_Issuer_Issuers",
    "Euclidian_Issuer_English",
    "Ks_stats_Subject_Subjects",
    "Ks_stats_Subject_English",
    "Ks_stats_Issuer_Issuers",
    "Ks_stats_Issuer_English",
    "Kl_dist_Subject_Subjects",
    "Kl_dist_Subject_English",
    "Kl_dist_Issuer_Issuers",
    "Kl_dist_Issuer_English",  # Ivan Torroledo's features
    "Length of domain",
    "Number of consecutive characters",
    "Entropy of domain",
    "Number of IP addresses",
    "Number of countries",
    "Average TTL value",
    "Standard deviation of TTL",
    "Life time of domain",
    "Active time of domain",  # Shi Yong's features
    "Domain name",
    "Entropy",
    "isDomainOnline",
    "diff ipList",
    "n WhoisAge",
    "diff Country",
    "mc Country",
    "diff City",
    "mc City",
    "TLD",
    "diff ASN",
    "mc ASN",
    "diff Organization",
    "mc Organization",
    "SubDomains",
    "UnusualChars",
    "Class",  # Magalhaes features
    "Domain length",
    "Domain Shannon entropy",
    "Domain pronunciation",
    "% of special symbols",
    "Number of NS records",
    "Registration Date",
    "Expiry Date",
    "Domains share the IP with",
    "Frequency of queries",  # Zhu Jiachen's features
    "Domain token count",
    "tld",
    "url Len",
    "domain length",
    "file",
    "Name Len",
    "dpath url Ratio",
    "Number of Dots in URL",
    "Query Digit Count",
    "Longest Path Token Length",
    "de-limiter Domain",
    "delimiter path",
    "Symbol Count Domain",
    "Entropy Domain",  # Kumar's features
    "nb days until collect",
    "nb days",
    "nb domain queries",
    "nb qnames",
    "min ttl",
    "ttl changes",
    "mean prot",
    "nb ips",
    "frequent aa",
    "frequent cd",
    "mean ancount",
    "mean arcount",
    "mean nscount",
    "frequent rcode",
    "mean qtype",
    "nb countries",
    "frequent country",
    "nb asns",
    "mean labels",
    "mean res len",  # Silveira's features
    "number of NS",
    "length of domain",
    "numeric chars",
    "mean TTL",
    "MX count",
    "Number of Vowels",
    "rate of numeric chars",
    "rate of vowels",
    "frequent cd",
    "mean ancount",
    "mean arcount",
    "mean nscount",
    "frequent rcode",
    "mean qtype",
    "nb countries",
    "frequent country",
    "nb asns",
    "mean labels",
    "mean res len",  # Iwahana's features
    "Autonomous System Number (ASN)",
    "Registrant",
    "Registrar",
    "Date",
    "IP addresses",
    "PTR record",
    "Global and Country ranking",
    "Webpages",
    "Time spent by visitor",
    "Web referrals",
    "Web traffic",
    "Category",
    "Geo-location",
    "Dots",
    "Underscores and hyphens",
    "Digits",
    "Illegitimate contents",  # Gopinath's features
    "Length of domain",
    "Number of consecutive characters",
    "Entropy of the domain",
    "Number of IP addresses",
    "Distinct geolocations of the IP addresses",
    "Mean TTL value",
    "Standard deviation of the TTL",
    "Life time of domain",
    "Active time of domain",  # Hason's features
    "Length of domain",
    "Number of consecutive characters",
    "Entropy of the domain",
    "Number of IP addresses",
    "Distinct geolocations of the IP addresses",
    "Mean TTL value",
    "Standard deviation of the TTL",
    "Life time of domain",
    "Active time of domain",  # Chatterjee's features
]

# Count the frequency of each feature
feature_count = Counter(features)

# Convert to DataFrame for easier visualization
df_feature_count = pd.DataFrame(feature_count.items(), columns=["Feature", "Frequency"])
df_feature_count = df_feature_count.sort_values(by="Frequency", ascending=False)

# Display the most common features
df_feature_count.head(20)

# Plot the most common features
plt.figure(figsize=(12, 8))
df_feature_count.head(20).plot(
    kind="barh", x="Feature", y="Frequency", legend=False, color="skyblue"
)
plt.title("Top 20 Most Popular Features")
plt.xlabel("Frequency")
plt.ylabel("Feature")
plt.gca().invert_yaxis()  # Invert y-axis for better visualization
plt.tight_layout()
plt.show()
