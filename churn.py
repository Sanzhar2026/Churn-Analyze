# ============================================
# CHURN PREDICTION PROJECT - ПОЛНАЯ МОДЕЛЬ
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve,
                             accuracy_score, f1_score, recall_score, precision_score)
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================

print("="*60)
print("CHURN PREDICTION PROJECT")
print("="*60)

# Загружаем ваш датасет
df = pd.read_csv('telco_churn.csv')  # ИЗМЕНИТЕ НАЗВАНИЕ ВАШЕГО ФАЙЛА

print(f"\n📊 Размер датасета: {df.shape}")
print(f"📊 Колонки: {df.columns.tolist()}")

# ============================================
# 2. ПРЕДОБРАБОТКА ДАННЫХ
# ============================================

def preprocess_data(df):
    """Полная предобработка данных"""
    
    data = df.copy()
    
    # Сохраняем целевую переменную отдельно
    if 'Churn' in data.columns:
        y_temp = data['Churn'].copy()
        data = data.drop('Churn', axis=1)
    else:
        y_temp = None
    
    # Удаляем ненужные колонки
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
    if 'CustomerID' in data.columns:
        data = data.drop('CustomerID', axis=1)
    
    # Преобразуем TotalCharges в числовой формат
    if 'TotalCharges' in data.columns:
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
    
    # Кодируем бинарные признаки
    binary_cols = ['Gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in data.columns:
            data[col] = data[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
    
    # One-hot encoding для мультикатегориальных признаков
    multi_cat_cols = ['InternetService', 'Contract', 'PaymentMethod', 
                      'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                      'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    for col in multi_cat_cols:
        if col in data.columns:
            # Заменяем 'No phone/internet service' на 'No' для упрощения
            if col in ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
                data[col] = data[col].replace({'No phone service': 'No', 
                                                'No internet service': 'No'})
            
            # One-hot encoding
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            data = pd.concat([data, dummies], axis=1)
            data = data.drop(col, axis=1)
    
    # ============================================
    # 3. FEATURE ENGINEERING
    # ============================================
    
    if 'Tenure' in data.columns and 'TotalCharges' in data.columns:
        # Средний чек за месяц
        data['Avg_Monthly_Spend'] = data['TotalCharges'] / (data['Tenure'] + 1)
        
        # Отношение ежемесячного платежа к общему
        data['Monthly_to_Total_Ratio'] = data['MonthlyCharges'] / (data['TotalCharges'] + 1)
        
        # Квадрат tenure (нелинейные зависимости)
        data['Tenure_Squared'] = data['Tenure'] ** 2
        
        # Логарифм TotalCharges
        data['Log_TotalCharges'] = np.log1p(data['TotalCharges'])
        
        # Группа по длительности обслуживания
        data['Tenure_Group'] = pd.cut(data['Tenure'], 
                                       bins=[-1, 6, 12, 24, 1000], 
                                       labels=['0-6 months', '6-12 months', '1-2 years', '2+ years'])
        
        # One-hot encoding для Tenure_Group
        tenure_dummies = pd.get_dummies(data['Tenure_Group'], prefix='Tenure', drop_first=True)
        data = pd.concat([data, tenure_dummies], axis=1)
        data = data.drop('Tenure_Group', axis=1)
        
        # Платит ли больше среднего
        if 'MonthlyCharges' in data.columns:
            avg_monthly = data['MonthlyCharges'].mean()
            data['Above_Avg_Monthly'] = (data['MonthlyCharges'] > avg_monthly).astype(int)
    
    # Заполняем оставшиеся NaN
    data = data.fillna(0)
    
    # Возвращаем целевую переменную
    if y_temp is not None:
        # Убеждаемся что y_temp не содержит NaN
        if y_temp.isna().any():
            nan_indices = y_temp[y_temp.isna()].index
            data = data.drop(nan_indices)
            y_temp = y_temp.drop(nan_indices)
        
        # Преобразуем в int
        y_temp = y_temp.astype(int)
        data['Churn'] = y_temp.values
    
    return data

# Применяем предобработку
df_processed = preprocess_data(df)

print(f"\n✅ После предобработки: {df_processed.shape[1]} признаков")
print(f"✅ NaN в данных: {df_processed.isna().sum().sum()}")

# ============================================
# 4. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ
# ============================================

X = df_processed.drop('Churn', axis=1)
y = df_processed['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Размер обучающей выборки: {X_train.shape}")
print(f"📊 Размер тестовой выборки: {X_test.shape}")
print(f"📊 Целевая в обучающей: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"📊 Целевая в тестовой: {dict(zip(*np.unique(y_test, return_counts=True)))}")

# ============================================
# 5. МАСШТАБИРОВАНИЕ И БАЛАНСИРОВКА
# ============================================

# Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Балансировка SMOTE
print("\n⚖️ Применяем SMOTE для балансировки классов...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"✅ После SMOTE: {dict(zip(*np.unique(y_train_resampled, return_counts=True)))}")

# ============================================
# 6. ОБУЧЕНИЕ МОДЕЛЕЙ
# ============================================

print("\n" + "="*60)
print("ОБУЧЕНИЕ МОДЕЛЕЙ")
print("="*60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
}

results = {}
best_model = None
best_f1 = 0
best_name = ""

for name, model in models.items():
    print(f"\n🔄 Обучение {name}...")
    
    # Обучаем
    model.fit(X_train_resampled, y_train_resampled)
    
    # Предсказания
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUC': auc,
        'Model': model
    }
    
    print(f"\n📈 {name}:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.1f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.1f}%)")
    print(f"  F1:        {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"    TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
    print(f"    FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name = name

print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best_name} с F1 = {best_f1:.4f}")

# ============================================
# 7. ОПТИМИЗАЦИЯ ПОРОГА РЕШЕНИЯ
# ============================================

print("\n" + "="*60)
print("ОПТИМИЗАЦИЯ ПОРОГА РЕШЕНИЯ")
print("="*60)

# Получаем вероятности для лучшей модели
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# Анализируем разные пороги
thresholds = np.arange(0.2, 0.8, 0.05)
best_threshold = 0.5
best_f1_thresh = 0

print("\nАнализ порогов решения:")
for thresh in thresholds:
    y_pred_thresh = (y_proba > thresh).astype(int)
    f1 = f1_score(y_test, y_pred_thresh)
    recall = recall_score(y_test, y_pred_thresh)
    precision = precision_score(y_test, y_pred_thresh)
    
    print(f"  Порог {thresh:.2f}: F1={f1:.4f}, Recall={recall:.4f}, Precision={precision:.4f}")
    
    if f1 > best_f1_thresh:
        best_f1_thresh = f1
        best_threshold = thresh

print(f"\n✅ ОПТИМАЛЬНЫЙ ПОРОГ: {best_threshold:.2f} (F1 = {best_f1_thresh:.4f})")

# Финальные предсказания с оптимальным порогом
y_pred_final = (y_proba > best_threshold).astype(int)

print(f"\n🎯 ФИНАЛЬНЫЕ МЕТРИКИ с порогом {best_threshold:.2f}:")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_final):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_final):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_final):.4f}")
print(f"  F1:        {f1_score(y_test, y_pred_final):.4f}")

# ============================================
# 8. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. ROC Curve
ax = axes[0, 0]
for name, result in results.items():
    model = result['Model']
    y_proba_model = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba_model)
    auc = result['AUC']
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves Comparison')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# 2. Feature Importance
ax = axes[0, 1]
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[-10:]
    feature_names = X.columns[indices]
    
    ax.barh(range(len(indices)), importances[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('Importance')
    ax.set_title(f'Top 10 Features ({best_name})')
elif hasattr(best_model, 'coef_'):
    importances = np.abs(best_model.coef_[0])
    indices = np.argsort(importances)[-10:]
    feature_names = X.columns[indices]
    
    ax.barh(range(len(indices)), importances[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('Coefficient Magnitude')
    ax.set_title(f'Top 10 Features ({best_name})')

# 3. Confusion Matrix
ax = axes[1, 0]
cm = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Stayed', 'Churned'],
            yticklabels=['Stayed', 'Churned'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix (Threshold={best_threshold:.2f})')

# 4. Metrics comparison
ax = axes[1, 1]
metrics_df = pd.DataFrame(results).T[['Accuracy', 'Precision', 'Recall', 'F1']]
metrics_df.plot(kind='bar', ax=ax)
ax.set_title('Models Comparison')
ax.set_ylabel('Score')
ax.set_xlabel('Model')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.legend(loc='lower right')
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('churn_model_results.png', dpi=150)
plt.show()

# ============================================
# 9. БИЗНЕС-МЕТРИКИ (ROI)
# ============================================

print("\n" + "="*60)
print("💰 БИЗНЕС-МЕТРИКИ (ROI)")
print("="*60)

# Допустим:
AVG_CUSTOMER_VALUE = 1000  # Средняя прибыль с клиента
RETENTION_COST = 50        # Стоимость удержания
RETENTION_SUCCESS = 0.30    # 30% удержанных

# Расчет
cm = confusion_matrix(y_test, y_pred_final)
tn, fp, fn, tp = cm.ravel()

total_customers = len(y_test)
actual_churners = fn + tp
predicted_churners = tp + fp

# Без модели: пытаемся удержать всех
cost_without_model = total_customers * RETENTION_COST

# С моделью: тратим только на предсказанных уходящих
cost_with_model = predicted_churners * RETENTION_COST

# Удерживаем успешно только настоящих уходящих
saved_customers = tp * RETENTION_SUCCESS
revenue_saved = saved_customers * AVG_CUSTOMER_VALUE

# ROI
cost_saved = cost_without_model - cost_with_model
net_profit = revenue_saved - cost_with_model
roi_percent = (net_profit / cost_with_model) * 100 if cost_with_model > 0 else 0

print(f"\n📊 ИСХОДНЫЕ ДАННЫЕ:")
print(f"  Всего клиентов в тесте: {total_customers}")
print(f"  Реальный отток: {actual_churners} ({actual_churners/total_customers:.1%})")
print(f"  Предсказано оттока: {predicted_churners}")

print(f"\n💵 ЗАТРАТЫ:")
print(f"  Без модели (удерживать всех): ${cost_without_model:,.0f}")
print(f"  С моделью (только рисковых): ${cost_with_model:,.0f}")
print(f"  Экономия на затратах: ${cost_saved:,.0f}")

print(f"\n💰 ДОХОД:")
print(f"  Спасено клиентов: {saved_customers:.0f}")
print(f"  Сохраненная выручка: ${revenue_saved:,.0f}")

print(f"\n📈 ROI:")
print(f"  Чистая прибыль: ${net_profit:,.0f}")
print(f"  ROI: {roi_percent:.1f}%")

# ============================================
# 10. СОХРАНЕНИЕ МОДЕЛИ (ОЧИЩЕННОЙ ОТ ОБЪЕКТОВ)
# ============================================

print("\n" + "="*60)
print("💾 СОХРАНЕНИЕ МОДЕЛИ")
print("="*60)

# ВАЖНО: Очищаем results от объектов моделей для JSON-безопасности
clean_results = {}
for name, metrics in results.items():
    clean_results[name] = {
        'Accuracy': metrics['Accuracy'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'F1': metrics['F1'],
        'AUC': metrics['AUC']
    }
    print(f"  ✅ Очищены метрики для {name}")

# Сохраняем лучшую модель и все компоненты
model_artifacts = {
    'model': best_model,  # Это объект модели, но joblib его сохранит
    'scaler': scaler,
    'feature_names': X.columns.tolist(),
    'best_threshold': best_threshold,
    'results': clean_results,  # ← ИСПОЛЬЗУЕМ ОЧИЩЕННЫЕ results (только числа)
    'avg_customer_value': AVG_CUSTOMER_VALUE,
    'retention_cost': RETENTION_COST,
    'retention_success': RETENTION_SUCCESS
}

# Сохраняем
joblib.dump(model_artifacts, 'churn_model_complete.pkl')
print(f"\n✅ Модель сохранена: 'churn_model_complete.pkl'")
print(f"✅ Results очищены от объектов моделей")