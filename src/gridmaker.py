import pandas as pd
import os

# Cesta ke složce
input_folder = "./grids/"

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
    output.append(r"  \small")  # trochu menší font pro úsporu místa
    output.append(f"  \\caption{{Výsledky pro subset {subset_label} – {attack_type}}}")
    output.append(r"  \begin{tabular}{|l|c|c|c|c|c|c|c|c|}")
    output.append(r"    \hline")
    output.append(
        r"    \textbf{Model} & \textbf{Acc} & \textbf{AUC} & \textbf{Recall} & \textbf{Prec.} & \textbf{F1} & \textbf{Kappa} & \textbf{MCC} & \textbf{TT (s)} \\"
    )
    output.append(r"    \hline")

    # Pokud tam jsou výsledky, napiš je
    if not df.empty:
        for _, row in df.iterrows():
            line = (
                f"    {row['Model']} & "
                f"{row['Accuracy']:.4f} & "
                f"{row['AUC']:.4f} & "
                f"{row['Recall']:.4f} & "
                f"{row['Prec.']:.4f} & "
                f"{row['F1']:.4f} & "
                f"{row['Kappa']:.4f} & "
                f"{row['MCC']:.4f} & "
                f"{row['TT (Sec)']:.2f} \\\\"
            )
            output.append(line)
    else:
        output.append(r"    \multicolumn{9}{c}{Žádná data} \\")

    output.append(r"    \hline")
    output.append(r"  \end{tabular}")
    output.append(r"\end{table}")
    output.append(r"\vspace{0.5cm}")  # malá mezera mezi tabulkami
    output.append("")  # prázdný řádek mezi tabulkami

# Ulož LaTeX soubor
with open("full_tables.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print("✅ Hotovo! Vygenerován soubor full_tables.tex")
