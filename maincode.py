import pandas as pd
import numpy as np
import hashlib
import json
import time

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from cryptography.fernet import Fernet

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

# LOAD DATA
dataset_path = "bank_transactions_data_2.csv"

df_original = pd.read_csv(dataset_path)

# HASH CALCULATION

def calculate_file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

file_hash = calculate_file_hash(dataset_path)

print("\nFile Hash:")
print(file_hash)

data_hash = hashlib.sha256(
    df_original.to_csv(index=False).encode()
).hexdigest()

print("\nData Hash:")
print(data_hash)


df = df_original.copy()

# DATE PROCESSING
date_columns = [
    "TransactionDate",
    "PreviousTransactionDate"
]

for col in date_columns:

    if col in df.columns:

        df[col] = pd.to_datetime(
            df[col],
            errors="coerce"
        )

if (
    "TransactionDate" in df.columns
    and
    "PreviousTransactionDate" in df.columns
):

    df["DaysGap"] = (
        df["TransactionDate"]
        -
        df["PreviousTransactionDate"]
    ).dt.days

for col in date_columns:

    if col in df.columns:

        df[col] = (
            df[col]
            .astype("int64")
            // 10**9
        )


# ENCODE ALL STRING COLUMNS
encoders = {}

for col in df.columns:

    if df[col].dtype == object:

        le = LabelEncoder()

        df[col] = le.fit_transform(
            df[col].astype(str)
        )

        encoders[col] = le


# ATTACK FLAG
if "TransactionAmount" in df.columns:

    amount_threshold = df[
        "TransactionAmount"
    ].quantile(0.95)

else:

    amount_threshold = 0

if "LoginAttempts" in df.columns:

    attack_login = df["LoginAttempts"] > 3

else:

    attack_login = False

if "TransactionAmount" in df.columns:

    attack_amount = (
        df["TransactionAmount"]
        >
        amount_threshold
    )

else:

    attack_amount = False

df["AttackFlag"] = (
    attack_amount |
    attack_login
).astype(int)

print("\nAttackFlag Counts")
print(df["AttackFlag"].value_counts())

# NEGLECTED FLAG
df["NeglectedFlag"] = (
    df.isnull().any(axis=1)
).astype(int)

print("\nNeglectedFlag Counts")
print(df["NeglectedFlag"].value_counts())


for col in df.columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    df[col].fillna(
        df[col].mean(),
        inplace=True
    )

# FEATURES
X = df.drop(
    columns=["AttackFlag"]
)

y = df["AttackFlag"]


scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train.values.reshape(-1, 1),
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test.values.reshape(-1, 1),
    dtype=torch.float32
)


# OLRPR MODEL
class OLRPR_Model(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),

            nn.Sigmoid()
        )

    def forward(self, x):

        return self.net(x)

# LION OPTIMIZER
class Lion(Optimizer):

    def __init__(
        self,
        params,
        lr=0.001,
        beta1=0.9,
        beta2=0.99
    ):

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2
        )

        super().__init__(
            params,
            defaults
        )

    @torch.no_grad()
    def step(self, closure=None):

        for group in self.param_groups:

            lr = group["lr"]
            beta2 = group["beta2"]

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]

                if "exp_avg" not in state:

                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                exp_avg.mul_(beta2).add_(
                    grad,
                    alpha=(1 - beta2)
                )

                update = exp_avg.sign()

                p.add_(
                    update,
                    alpha=-lr
                )


# TRAINING
model = OLRPR_Model(
    X_train.shape[1]
)

criterion = nn.BCELoss()

optimizer = Lion(
    model.parameters(),
    lr=0.01
)

epochs = 500

for epoch in range(epochs):

    model.train()

    outputs = model(X_train)

    loss = criterion(
        outputs,
        y_train
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 50 == 0:

        pred = (
            outputs > 0.5
        ).float()

        acc = (
            pred == y_train
        ).float().mean()

        print(
            f"Epoch {epoch+1}/{epochs}"
            f" Loss={loss.item():.4f}"
            f" Accuracy={acc.item()*100:.2f}%"
        )

# TESTING
model.eval()

with torch.no_grad():

    test_output = model(X_test)

    y_pred = (
        test_output > 0.5
    ).int()

acc = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy")
print(acc * 100)

print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# ENCRYPTION
key = Fernet.generate_key()

cipher = Fernet(key)

encrypted_rows = []

hash2_list = []

for row in X_scaled:

    encrypted = cipher.encrypt(
        json.dumps(
            row.tolist()
        ).encode()
    )

    encrypted_rows.append(encrypted)

    hash2 = hashlib.sha256(
        encrypted
    ).hexdigest()

    hash2_list.append(hash2)

encrypted_df = pd.DataFrame({

    "EncryptedData":
    encrypted_rows,

    "Hash2":
    hash2_list,

    "Target":
    y.values
})

print("\nEncrypted Data Sample")
print(encrypted_df.head())

# CLOUD AUDIT
audit_log = []

def cloud_audit(
    event,
    details
):

    audit_log.append({

        "timestamp":
        time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "event":
        event,

        "details":
        details
    })

cloud_audit(
    "ENCRYPTION",
    "Data encrypted"
)

cloud_audit(
    "TRAINING",
    "Model trained"
)

cloud_audit(
    "PREDICTION",
    "Prediction completed"
)

print("\nAudit Logs")

for log in audit_log:

    print(
        f"[{log['timestamp']}] "
        f"{log['event']} "
        f"{log['details']}"
    )

# ATTACK RECORDS
print("\nAttack Records")
print(
    df[
        df["AttackFlag"] == 1
    ].head()
)

print("\nNeglected Records")
print(
    df[
        df["NeglectedFlag"] == 1
    ].head()
)