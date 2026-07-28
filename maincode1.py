import pandas as pd
import numpy as np
import hashlib
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and calculate SHA-256 hash
cloud_path = "archive/HepatitisCdata.csv"

def calculate_file_hash(filepath):
    with open(filepath, "rb") as f:
        file_bytes = f.read()
        return hashlib.sha256(file_bytes).hexdigest()

file_hash = calculate_file_hash(cloud_path)
print(" File SHA-256 Hash:", file_hash)

df_original = pd.read_csv(cloud_path)

def calculate_dataframe_hash(df):
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()

raw_data_hash = calculate_dataframe_hash(df_original)
print(" Original DataFrame Hash:", raw_data_hash)

# Step 2: Clean & Encode
df = df_original.copy()
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

# Keep only Blood Donor and Suspect Blood Donor categories
df = df[df['Category'].isin(['0=Blood Donor', '0s=suspect Blood Donor'])]

# Binary Encoding: 0 = Blood Donor, 1 = Suspect
df['Category'] = df['Category'].map({'0=Blood Donor': 0, '0s=suspect Blood Donor': 1})

# Step 3: Attack flag based on abnormal enzymes
df['AttackFlag'] = ((df['ALT'] > 100) | (df['AST'] > 100) | (df['ALP'] > 300)).astype(int)
print("\n AttackFlag Counts:\n", df['AttackFlag'].value_counts())

# Step 4: Handle missing values
df['NeglectedFlag'] = df.isnull().any(axis=1).astype(int)
print("\n NeglectedFlag Counts:\n", df['NeglectedFlag'].value_counts())

df.fillna(df.mean(numeric_only=True), inplace=True)

# Step 5: Prepare data
X = df.drop(columns=['Category'])
y = df['Category']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
y_test = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

# Step 6: Define OLRPR Model
class OLRPR_Model(nn.Module):
    def __init__(self, input_size):
        super(OLRPR_Model, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Step 7: Lion Optimizer
class Lion(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.99):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2)
        super(Lion, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
                update = exp_avg.sign()
                p.add_(update, alpha=-lr)

# Step 8: Train Model
model = OLRPR_Model(input_size=X_train.shape[1])
criterion = nn.BCELoss()
optimizer = Lion(model.parameters(), lr=0.01)
epochs = 500

for epoch in range(epochs):
    model.train()
    output = model(X_train)
    loss = criterion(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        preds = (output > 0.5).float()
        acc = (preds == y_train).float().mean().item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Accuracy: {acc*100:.2f}%")

# Step 9: Evaluate Model
model.eval()
with torch.no_grad():
    test_output = model(X_test)
    y_pred = (test_output > 0.5).int()
    y_true = y_test.int()

print(f"\n Final Test Accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%")

print("\n Classification Report:")
print(classification_report(
    y_true, y_pred,
    target_names=["Blood Donor", "Suspect Donor"],
    digits=3
))

cm = confusion_matrix(y_true, y_pred)
print("\n Confusion Matrix:\n", cm)

# Plot
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Blood Donor", "Suspect Donor"],
            yticklabels=["Blood Donor", "Suspect Donor"])
plt.title("OLRPR Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

#  Extra: Show sample of attack predictions and neglected records
print("\n Sample Attack Predicted Records:")
print(df[df['AttackFlag'] == 1].head())

print("\n Sample Neglected Entries (originally had NaNs):")
print(df[df['NeglectedFlag'] == 1].head())

import pandas as pd
import numpy as np
from cryptography.fernet import Fernet
import hashlib
import base64

# Load dataset
df = pd.read_csv('C:/Users/saini/OneDrive/Desktop/Cloud Computing/archive/HepatitisCdata.csv')

# Encode categorical features
df['Sex'] = df['Sex'].astype(str)
df['Category'] = df['Category'].astype(str)
df['Sex'] = df['Sex'].map({'m': 1, 'f': 0})
df['Category'] = df['Category'].astype('category').cat.codes

# Filter classes: 0 = Blood Donor, 1 = Suspect Blood Donor
df = df[df['Category'].isin([0, 1])]
df.fillna(df.mean(numeric_only=True), inplace=True)

X = df.drop(columns=['Category'])
y = df['Category']

# Normalize features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

#  Step 1: Encrypt the normalized feature data using Fernet
key = Fernet.generate_key()
cipher = Fernet(key)

encrypted_rows = []
for row in X_scaled:
    # Convert row to string of bytes
    row_bytes = str(row.tolist()).encode()
    encrypted = cipher.encrypt(row_bytes)
    encrypted_rows.append(encrypted)

#  Step 2: Hash 2 Calculation using SHA-256
hash2_list = []
for encrypted_row in encrypted_rows:
    hash2 = hashlib.sha256(encrypted_row).hexdigest()
    hash2_list.append(hash2)

# Convert to DataFrame
df_encrypted = pd.DataFrame({'Encrypted': encrypted_rows, 'Hash2': hash2_list})
df_encrypted['Target'] = y.values

print(" Sample Encrypted + Hash2 Data:")
print(df_encrypted.head())

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from cryptography.fernet import Fernet
import hashlib
import time
import json

# Load and preprocess
df = pd.read_csv('C:/Users/saini/OneDrive/Desktop/Cloud Computing/archive/HepatitisCdata.csv')
df['Sex'] = df['Sex'].map({'m': 1, 'f': 0})
df['Category'] = df['Category'].astype('category').cat.codes
df = df[df['Category'].isin([0, 1])]
df.fillna(df.mean(numeric_only=True), inplace=True)

X = df.drop(columns=['Category'])
y = df['Category']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

#  Key generation
key = Fernet.generate_key()
cipher = Fernet(key)

# Simulate Homomorphic Encryption (encrypting features)
def encrypt_data(X_scaled):
    encrypted_data = []
    for row in X_scaled:
        row_str = json.dumps(row.tolist())  # serialize
        encrypted = cipher.encrypt(row_str.encode())
        encrypted_data.append(encrypted)
    return encrypted_data

#  Simulated Cloud Auditing
audit_log = []

def cloud_audit(event_type, details):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {"timestamp": timestamp, "event": event_type, "details": details}
    audit_log.append(log_entry)

#  Encrypt
cloud_audit("ENCRYPTION", "User data encrypted before upload")
encrypted_data = encrypt_data(X_scaled)

#  Simulate "Cloud" Decryption for model usage (only for authenticated users)
def decrypt_data(encrypted_data, key):
    cipher = Fernet(key)
    decrypted_rows = []
    for enc_row in encrypted_data:
        decrypted = cipher.decrypt(enc_row)
        row = np.array(json.loads(decrypted)).astype(np.float32)
        decrypted_rows.append(row)
    return np.array(decrypted_rows)

#  Authenticated user access
user_authenticated = True

if user_authenticated:
    cloud_audit("AUTH_SUCCESS", "User authenticated for decryption")
    decrypted_X = decrypt_data(encrypted_data, key)

    # Model Training
    X_train, X_test, y_train, y_test = train_test_split(decrypted_X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)

    cloud_audit("PREDICTION", "Model made prediction on decrypted data")


else:
    cloud_audit("AUTH_FAIL", "Unauthorized decryption attempt blocked")
    print(" Access denied. Authentication failed.")

# Show Audit Logs
print("\n Cloud Audit Log:")
for entry in audit_log:
    print(f"[{entry['timestamp']}] {entry['event']}: {entry['details']}")