import pandas as pd
import os

# Cesta ke složce
input_folder = "./grids/"

# Mapa zkratek pro modely
model_abbr = {
    "Random Forest Classifier": r"\texttt{RF}",
    "Ada Boost Classifier": r"\texttt{ADA}",
    "Decision Tree Classifier": r"\texttt{DT}",
    "Ridge Classifier": r"\texttt{Ridge}",
    "K Neighbors Classifier": r"\texttt{KNN}",
    "SVM - Linear Kernel": r"\texttt{SVM-L}",
    "Logistic Regression": r"\texttt{LR}",
    "Quadratic Discriminant Analysis": r"\texttt{QDA}",
    "Naive Bayes": r"\texttt{NB}",
    "Extra Trees Classifier": r"\texttt{ET}",
    "Extreme Gradient Boosting": r"\texttt{XGB}",
    "Light Gradient Boosting Machine": r"\texttt{LGBM}",
    "Gradient Boosting Classifier": r"\texttt{GBC}",
    "Linear Discriminant Analysis": r"\texttt{LDA}",
    "Dummy Classifier": r"\texttt{Dummy}",
}

# Všechny CSV soubory ve složce
csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

# Výstupní seznam LaTeX řádků
output = []

for filename in sorted(csv_files):
    filepath = os.path.join(input_folder, filename)

    # Načti CSV
    df = pd.read_csv(filepath)

    # Urči typ útoku (malware/phishing) ze jména souboru
    if filename.startswith("malware"):
        attack_type = "malware"
        subset_name = (
            filename[len("malware") :]
            .replace("__agg", "")
            .replace(".csv", "")
            .strip("_")
        )
    elif filename.startswith("phishing"):
        attack_type = "phishing"
        subset_name = (
            filename[len("phishing") :]
            .replace("__agg", "")
            .replace(".csv", "")
            .strip("_")
        )
    else:
        print(f"⚠️ Neznámý typ souboru: {filename}")
        continue

    # Nahraď některé složité znaky pro lepší LaTeX výpis
    subset_label = subset_name.replace("+", "+").replace("_", "")

    # Přidej tabulku do výstupu
    output.append(r"\begin{table}[H]")
    output.append(r"  \centering")
    output.append(r"  \small")
    output.append(f"  \\caption{{Výsledky pro subset {subset_label} – {attack_type}}}")
    output.append(r"  \begin{tabular}{|l|c|c|c|c|c|c|c|}")
    output.append(r"    \hline")
    output.append(
        r"    \textbf{Model} & \textbf{Acc} & \textbf{AUC} & \textbf{Recall} & \textbf{Prec.} & \textbf{F1} & \textbf{Kappa} & \textbf{MCC} \\"
    )
    output.append(r"    \hline")

    if not df.empty:
        for _, row in df.iterrows():
            model_full = str(row["Model"]).strip()
            model_short = model_abbr.get(
                model_full, model_full
            )  # fallback na původní název
            line = (
                f"    {model_short} & "
                f"{row['Accuracy']:.4f} & "
                f"{row['AUC']:.4f} & "
                f"{row['Recall']:.4f} & "
                f"{row['Prec.']:.4f} & "
                f"{row['F1']:.4f} & "
                f"{row['Kappa']:.4f} & "
                f"{row['MCC']:.4f} \\\\"
            )
            output.append(line)
    else:
        output.append(r"    \multicolumn{8}{c}{Žádná data} \\")

    output.append(r"    \hline")
    output.append(r"  \end{tabular}")
    output.append(r"\end{table}")
    output.append(r"\vspace{0.5cm}")
    output.append("")

# Ulož LaTeX soubor
with open("full_tables.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print("✅ Hotovo! Vygenerován soubor full_tables.tex")
