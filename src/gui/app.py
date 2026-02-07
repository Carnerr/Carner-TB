from __future__ import annotations

import subprocess
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import pandas as pd


class TrainerGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Carner-TB Trading Bot Control Panel")
        self.geometry("900x600")

        self.output = tk.Text(self, wrap="word", height=16)
        self.output.pack(fill="both", expand=False, padx=8, pady=8)

        button_frame = ttk.Frame(self)
        button_frame.pack(fill="x", padx=8)

        ttk.Button(button_frame, text="Pull Alpha Vantage Data", command=self.run_data_pull).pack(side="left", padx=4)
        ttk.Button(button_frame, text="Update Data", command=self.run_data_update).pack(side="left", padx=4)
        ttk.Button(button_frame, text="Run Baseline", command=self.run_baseline).pack(side="left", padx=4)
        ttk.Button(button_frame, text="Train Model", command=self.run_training).pack(side="left", padx=4)
        ttk.Button(button_frame, text="Organize Data", command=self.run_data_organize).pack(side="left", padx=4)
        ttk.Button(button_frame, text="Refresh Metrics", command=self.refresh_metrics).pack(side="left", padx=4)

        self.metrics_table = ttk.Treeview(self, columns=("episode", "epsilon", "avg_reward", "avg_loss", "algo"), show="headings")
        for col in ("episode", "epsilon", "avg_reward", "avg_loss", "algo"):
            self.metrics_table.heading(col, text=col)
        self.metrics_table.pack(fill="both", expand=True, padx=8, pady=8)

    def _append_output(self, text: str) -> None:
        self.output.insert("end", text + "\n")
        self.output.see("end")

    def _run_command(self, command: list[str]) -> None:
        self._append_output(f"Running: {' '.join(command)}")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if process.stdout:
            for line in process.stdout:
                self._append_output(line.rstrip())
        process.wait()
        self._append_output(f"Completed with code {process.returncode}")

    def run_data_pull(self) -> None:
        threading.Thread(target=self._run_command, args=(["python", "-m", "src.data.alpha_vantage"],), daemon=True).start()

    def run_training(self) -> None:
        threading.Thread(target=self._run_command, args=(["python", "-m", "src.rl.train"],), daemon=True).start()

    def run_baseline(self) -> None:
        threading.Thread(target=self._run_command, args=(["python", "-m", "src.analysis.baseline"],), daemon=True).start()

    def run_data_update(self) -> None:
        threading.Thread(
            target=self._run_command,
            args=(["python", "-c", "from src.data.alpha_vantage import update; from src.config import AlphaVantageConfig; update(AlphaVantageConfig())"],),
            daemon=True,
        ).start()

    def run_data_organize(self) -> None:
        threading.Thread(target=self._run_command, args=(["python", "-m", "src.data.organize"],), daemon=True).start()

    def refresh_metrics(self) -> None:
        metrics_path = Path("logs/training_metrics.csv")
        if not metrics_path.exists():
            self._append_output("No metrics found. Run training first.")
            return
        df = pd.read_csv(metrics_path)
        for row in self.metrics_table.get_children():
            self.metrics_table.delete(row)
        for _, row in df.tail(20).iterrows():
            self.metrics_table.insert(
                "",
                "end",
                values=(
                    int(row["episode"]),
                    f"{float(row['epsilon']):.4f}",
                    f"{float(row['avg_reward']):.2f}",
                    f"{float(row['avg_loss']):.4f}",
                    row["algo"],
                ),
            )


if __name__ == "__main__":
    app = TrainerGUI()
    app.mainloop()
