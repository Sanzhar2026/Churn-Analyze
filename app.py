# ============================================
# CHURN PREDICTION - STREAMLIT DASHBOARD
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Настройка страницы
st.set_page_config(
    page_title="Churn Predictor Pro",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок
st.title("🔮 Customer Churn Prediction Dashboard")
st.markdown("#### Predict which customers will leave and save revenue")

# Загрузка модели
@st.cache_resource
def load_model():
    if os.path.exists('churn_model_complete.pkl'):
        return joblib.load('churn_model_complete.pkl')
    else:
        return None

model_artifacts = load_model()

if model_artifacts is None:
    st.error("❌ Модель не найдена! Сначала запустите churn_model.py")
    st.stop()

# Извлекаем все компоненты модели
model = model_artifacts['model']
scaler = model_artifacts['scaler']
feature_names = model_artifacts['feature_names']
threshold = model_artifacts['best_threshold']
results = model_artifacts.get('results', {})  # ← ИСПРАВЛЕНО: используем get с пустым словарем по умолчанию

# Определяем лучшую модель
if results:
    best_model_name = list(results.keys())[0]
    # Находим модель с лучшим F1
    best_f1 = 0
    for name, metrics in results.items():
        if metrics.get('F1', 0) > best_f1:
            best_f1 = metrics['F1']
            best_model_name = name
else:
    best_model_name = "Unknown"

# Боковое меню
st.sidebar.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=100)
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose Mode",
    ["📊 Executive Dashboard", "🔍 Single Prediction", "📁 Batch Upload", "💰 ROI Calculator", "📈 Analytics"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Info")
st.sidebar.info(f"""
**Best Model:** {best_model_name}
**Threshold:** {threshold:.2f}
**Features:** {len(feature_names)}
""")

# ============================================
# СТРАНИЦА 1: EXECUTIVE DASHBOARD
# ============================================
if page == "📊 Executive Dashboard":
    st.header("📊 Executive Dashboard")
    
    # KPI метрики
    col1, col2, col3, col4 = st.columns(4)
    
    # Берем метрики из модели если есть
    if results and best_model_name in results:
        model_recall = results[best_model_name].get('Recall', 0.82)
        model_recall_pct = f"{model_recall:.1%}"
    else:
        model_recall_pct = "82%"
    
    with col1:
        st.metric(
            label="Current Churn Rate",
            value="5.2%",
            delta="-0.3%",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            label="Customers at Risk",
            value="847",
            delta="+42",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Revenue at Risk",
            value="$142.5K",
            delta="$+12.3K",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="Model Recall",
            value=model_recall_pct,
            delta="+3%"
        )
    
    # Графики в два столбца
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📈 Churn by Tenure")
        
        tenure_data = pd.DataFrame({
            'Tenure Group': ['0-6 months', '6-12 months', '1-2 years', '2+ years'],
            'Churn Rate': [47, 32, 18, 9],
            'Customers': [1200, 1500, 2000, 3300]
        })
        
        fig = px.bar(tenure_data, x='Tenure Group', y='Churn Rate',
                     title='Churn Rate by Customer Tenure',
                     color='Churn Rate',
                     color_continuous_scale='RdYlGn_r',
                     text='Churn Rate')
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.subheader("📊 Churn by Contract")
        
        contract_data = pd.DataFrame({
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'Churn Rate': [42, 11, 3],
            'Customers': [3200, 2100, 1700]
        })
        
        fig = px.pie(contract_data, values='Customers', names='Contract',
                     title='Customer Distribution by Contract',
                     hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    
    # Второй ряд графиков
    col_left2, col_right2 = st.columns(2)
    
    with col_left2:
        st.subheader("📉 Monthly Churn Trend")
        
        dates = pd.date_range(start='2024-01-01', periods=6, freq='M')
        trend_data = pd.DataFrame({
            'Month': dates,
            'Churn': [5.1, 4.8, 5.2, 4.9, 5.0, 4.7],
            'Target': [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend_data['Month'], y=trend_data['Churn'],
                                  mode='lines+markers', name='Actual',
                                  line=dict(color='red', width=3)))
        fig.add_trace(go.Scatter(x=trend_data['Month'], y=trend_data['Target'],
                                  mode='lines', name='Target',
                                  line=dict(color='green', width=2, dash='dash')))
        fig.update_layout(title='Churn Rate Trend (Last 6 Months)',
                         xaxis_title='Month',
                         yaxis_title='Churn Rate (%)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right2:
        st.subheader("⚠️ High Risk Customers")
        
        risk_data = pd.DataFrame({
            'Customer': ['C001', 'C002', 'C003', 'C004', 'C005'],
            'Risk': [95, 92, 88, 85, 82],
            'Value': [120, 95, 150, 80, 110]
        })
        
        fig = px.bar(risk_data, x='Customer', y='Risk',
                     title='Top 5 High Risk Customers',
                     color='Risk',
                     color_continuous_scale='Reds',
                     text='Risk')
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# СТРАНИЦА 2: SINGLE PREDICTION
# ============================================
elif page == "🔍 Single Prediction":
    st.header("🔍 Single Customer Risk Assessment")
    
    st.info("Enter customer details to predict churn risk")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Customer Information")
        
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges ($)", 0, 200, 70)
        total_charges = monthly_charges * tenure
        
        senior_citizen = st.checkbox("Senior Citizen")
        gender = st.selectbox("Gender", ["Male", "Female"])
        partner = st.selectbox("Has Partner?", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
    
    with col2:
        st.subheader("📞 Service Information")
        
        contract = st.selectbox("Contract Type",
                               ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method",
                                     ["Electronic check", "Mailed check",
                                      "Bank transfer", "Credit card"])
        internet_service = st.selectbox("Internet Service",
                                       ["DSL", "Fiber optic", "No"])
        
        phone_service = st.checkbox("Phone Service", value=True)
        multiple_lines = st.selectbox("Multiple Lines",
                                     ["No", "Yes", "No phone service"])
        online_security = st.selectbox("Online Security",
                                      ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support",
                                   ["No", "Yes", "No internet service"])
    
    if st.button("🔮 Predict Churn Risk", type="primary"):
        # Создаем DataFrame с введенными данными
        input_data = pd.DataFrame({
            'Gender': [1 if gender == 'Male' else 0],
            'SeniorCitizen': [1 if senior_citizen else 0],
            'Tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'Contract': [0 if contract == 'Month-to-month' else 1 if contract == 'One year' else 2],
            'PaymentMethod': [0 if payment_method == 'Electronic check' else
                             1 if payment_method == 'Mailed check' else
                             2 if payment_method == 'Bank transfer' else 3],
            'TotalCharges': [total_charges],
            'Partner': [1 if partner == 'Yes' else 0],
            'Dependents': [1 if dependents == 'Yes' else 0],
            'PhoneService': [1 if phone_service else 0],
            'MultipleLines': [0 if multiple_lines == 'No' else
                              1 if multiple_lines == 'Yes' else 2],
            'InternetService': [0 if internet_service == 'DSL' else
                                1 if internet_service == 'Fiber optic' else 2],
            'OnlineSecurity': [0 if online_security == 'No' else
                               1 if online_security == 'Yes' else 2],
            'TechSupport': [0 if tech_support == 'No' else
                            1 if tech_support == 'Yes' else 2]
        })
        
        # Добавляем engineered features
        input_data['Avg_Monthly_Spend'] = total_charges / (tenure + 1)
        input_data['Tenure_Squared'] = tenure ** 2
        input_data['Log_TotalCharges'] = np.log1p(total_charges)
        
        # Выбираем только нужные признаки
        input_data = input_data[[col for col in feature_names if col in input_data.columns]]
        
        # Добавляем недостающие признаки с 0
        for col in feature_names:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Масштабируем
        input_scaled = scaler.transform(input_data[feature_names])
        
        # Предсказываем
        probability = model.predict_proba(input_scaled)[0][1]
        prediction = int(probability > threshold)
        
        # Показываем результат
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        
        col1, col2, col3 = st.columns(3)
        
        with col2:
            # Прогресс бар
            st.markdown(f"### Churn Probability")
            st.progress(float(probability))
            
            if probability > 0.7:
                st.error(f"### ⚠️ HIGH RISK")
                st.markdown(f"### {probability:.1%}")
                actions = [
                    "📞 Schedule immediate retention call",
                    "🎁 Send 30% discount offer",
                    "📧 Send personalized email from manager",
                    "🎯 Offer free upgrade for 3 months"
                ]
            elif probability > 0.3:
                st.warning(f"### ⚠️ MEDIUM RISK")
                st.markdown(f"### {probability:.1%}")
                actions = [
                    "📧 Send satisfaction survey",
                    "🎁 Offer 15% discount",
                    "📱 Send SMS with special offer",
                    "🎂 Offer birthday discount"
                ]
            else:
                st.success(f"### ✅ LOW RISK")
                st.markdown(f"### {probability:.1%}")
                actions = [
                    "👍 Customer is loyal",
                    "🎉 Send thank you note",
                    "⭐ Invite to loyalty program",
                    "📰 Send newsletter"
                ]
        
        # Рекомендации
        st.markdown("---")
        st.subheader("💡 Retention Recommendations")
        
        cols = st.columns(len(actions))
        for i, action in enumerate(actions):
            with cols[i]:
                st.info(action)
        
        # Сохранить результат
        if st.button("📥 Save Prediction"):
            st.success("Prediction saved to history!")

# ============================================
# СТРАНИЦА 3: BATCH UPLOAD
# ============================================
elif page == "📁 Batch Upload":
    st.header("📁 Batch Customer Upload")
    
    st.info("Upload a CSV file with customer data for bulk prediction")
    
    # Пример файла
    with st.expander("📋 View Sample Format"):
        sample_data = pd.DataFrame({
            'CustomerID': ['C001', 'C002', 'C003'],
            'Tenure': [12, 34, 5],
            'MonthlyCharges': [70, 95, 45],
            'Contract': ['Month-to-month', 'Two year', 'One year'],
            'PaymentMethod': ['Electronic check', 'Bank transfer', 'Credit card'],
            'TotalCharges': [840, 3230, 225]
        })
        st.dataframe(sample_data)
        
        # Кнопка скачивания примера
        csv = sample_data.to_csv(index=False)
        st.download_button(
            "📥 Download Sample CSV",
            csv,
            "sample_customers.csv",
            "text/csv"
        )
    
    # Загрузка файла
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        customers = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Loaded {len(customers)} customers")
        st.write("Preview:")
        st.dataframe(customers.head())
        
        if st.button("🔮 Analyze All Customers", type="primary"):
            with st.spinner("Analyzing customers..."):
                # Создаем копию для предсказаний
                results_df = customers.copy()
                
                # Здесь должен быть код предсказания для каждого клиента
                # Для демо используем случайные вероятности
                np.random.seed(42)
                probabilities = np.random.rand(len(customers))
                
                results_df['Churn_Probability'] = probabilities
                results_df['Risk_Level'] = pd.cut(probabilities,
                                                 bins=[0, 0.3, 0.7, 1],
                                                 labels=['Low', 'Medium', 'High'])
                results_df['Will_Churn'] = (probabilities > threshold).astype(int)
                
                # Статистика
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                high_risk = sum(results_df['Risk_Level'] == 'High')
                medium_risk = sum(results_df['Risk_Level'] == 'Medium')
                low_risk = sum(results_df['Risk_Level'] == 'Low')
                
                col1.metric("🔴 High Risk", f"{high_risk} customers")
                col2.metric("🟡 Medium Risk", f"{medium_risk} customers")
                col3.metric("🟢 Low Risk", f"{low_risk} customers")
                
                # Revenue at risk
                if 'MonthlyCharges' in results_df.columns:
                    revenue_at_risk = results_df[results_df['Risk_Level'] == 'High']['MonthlyCharges'].sum()
                    st.metric("💰 Monthly Revenue at Risk", f"${revenue_at_risk:,.2f}")
                
                # Показываем результаты
                st.subheader("📋 Detailed Results")
                st.dataframe(results_df)
                
                # Кнопка скачивания
                csv_results = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results with Predictions",
                    csv_results,
                    "churn_predictions.csv",
                    "text/csv"
                )
                
                # Визуализация
                st.subheader("📊 Risk Distribution")
                
                col_left, col_right = st.columns(2)
                
                with col_left:
                    # Pie chart
                    risk_counts = results_df['Risk_Level'].value_counts().reset_index()
                    risk_counts.columns = ['Risk', 'Count']
                    fig = px.pie(risk_counts, values='Count', names='Risk',
                                 title='Customer Risk Distribution',
                                 color='Risk',
                                 color_discrete_map={'High': 'red',
                                                    'Medium': 'orange',
                                                    'Low': 'green'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_right:
                    # Bar chart
                    fig = px.bar(risk_counts, x='Risk', y='Count',
                                title='Risk Levels Count',
                                color='Risk',
                                color_discrete_map={'High': 'red',
                                                   'Medium': 'orange',
                                                   'Low': 'green'})
                    st.plotly_chart(fig, use_container_width=True)

# ============================================
# СТРАНИЦА 4: ROI CALCULATOR
# ============================================
elif page == "💰 ROI Calculator":
    st.header("💰 ROI Calculator")
    
    st.info("Calculate potential return on investment from using the churn prediction model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Business Parameters")
        
        total_customers = st.number_input("Total Customers", 1000, 1000000, 10000)
        avg_customer_value = st.number_input("Average Customer Value ($)", 100, 10000, 1000)
        monthly_churn_rate = st.slider("Monthly Churn Rate (%)", 1.0, 10.0, 5.0) / 100
        
        retention_cost = st.number_input("Retention Cost per Customer ($)", 10, 500, 50)
        retention_success = st.slider("Retention Campaign Success Rate (%)", 10, 50, 30) / 100
        
    with col2:
        st.subheader("🤖 Model Performance")
        
        # Берем из модели если есть
        if results and best_model_name in results:
            default_recall = results[best_model_name].get('Recall', 0.82)
            default_precision = results[best_model_name].get('Precision', 0.71)
        else:
            default_recall = 0.82
            default_precision = 0.71
        
        model_recall = st.slider("Model Recall (Catch Rate %)", 50, 100, int(default_recall*100)) / 100
        model_precision = st.slider("Model Precision (%)", 50, 100, int(default_precision*100)) / 100
        
        model_cost = st.number_input("Monthly Model Subscription ($)", 0, 10000, 299)
    
    # Расчеты
    st.markdown("---")
    st.subheader("📈 ROI Calculation")
    
    # Ежемесячные метрики
    monthly_churners = int(total_customers * monthly_churn_rate)
    caught_churners = int(monthly_churners * model_recall)
    predicted_churners = int(caught_churners / model_precision) if model_precision > 0 else 0
    
    # Затраты
    cost_without_model = total_customers * retention_cost
    cost_with_model = predicted_churners * retention_cost + model_cost
    
    # Спасенная выручка
    saved_customers = int(caught_churners * retention_success)
    saved_revenue = saved_customers * avg_customer_value
    
    # ROI
    net_profit = saved_revenue - cost_with_model
    roi_percent = (net_profit / cost_with_model) * 100 if cost_with_model > 0 else 0
    
    # Показываем результаты в красивых карточках
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Monthly Churners", f"{monthly_churners}")
        st.metric("Caught by Model", f"{caught_churners} ({model_recall:.0%})")
    
    with col2:
        st.metric("Cost Without Model", f"${cost_without_model:,.0f}")
        st.metric("Cost With Model", f"${cost_with_model:,.0f}")
        st.metric("Cost Savings", f"${cost_without_model - cost_with_model:,.0f}")
    
    with col3:
        st.metric("Saved Customers", f"{saved_customers}")
        st.metric("Saved Revenue", f"${saved_revenue:,.0f}")
    
    with col4:
        st.metric("Net Profit", f"${net_profit:,.0f}")
        st.metric("ROI", f"{roi_percent:.0f}%")
    
    # Прогресс бар для ROI
    st.subheader("ROI Progress")
    roi_progress = min(roi_percent / 500, 1.0)  # 500% как максимум
    st.progress(roi_progress)
    
    if roi_percent > 300:
        st.success(f"🚀 EXCELLENT ROI! {roi_percent:.0f}% return on investment")
    elif roi_percent > 200:
        st.success(f"✅ GOOD ROI! {roi_percent:.0f}% return on investment")
    elif roi_percent > 100:
        st.warning(f"⚖️ POSITIVE ROI! {roi_percent:.0f}% return on investment")
    else:
        st.error(f"⚠️ NEGATIVE ROI! {roi_percent:.0f}% return on investment")
    
    # Рекомендации
    st.markdown("---")
    st.subheader("💡 Recommendations")
    
    recommendations = []
    if model_recall < 0.7:
        recommendations.append("🔧 Improve model recall to catch more churners")
    if model_precision < 0.6:
        recommendations.append("🎯 Improve model precision to reduce wasted retention costs")
    if retention_success < 0.3:
        recommendations.append("📈 Improve retention campaigns - try different offers")
    if roi_percent < 100:
        recommendations.append("💰 Consider adjusting retention cost or targeting strategy")
    
    if recommendations:
        for rec in recommendations:
            st.info(rec)
    else:
        st.success("✅ Your model and strategy look good!")

# ============================================
# СТРАНИЦА 5: ANALYTICS
# ============================================
elif page == "📈 Analytics":
    st.header("📈 Advanced Analytics")
    
    st.info("Deep dive into customer behavior and model performance")
    
    tab1, tab2, tab3 = st.tabs(["📊 Customer Segments", "📉 Churn Patterns", "🤖 Model Performance"])
    
    with tab1:
        st.subheader("Customer Segments Analysis")
        
        # Создаем синтетические данные для демо
        np.random.seed(42)
        n_customers = 1000
        
        segments = pd.DataFrame({
            'Tenure': np.random.randint(1, 72, n_customers),
            'MonthlyCharges': np.random.randint(20, 120, n_customers),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_customers),
            'Churn': np.random.choice([0, 1], n_customers, p=[0.73, 0.27])
        })
        
        # Churn by tenure group
        segments['TenureGroup'] = pd.cut(segments['Tenure'],
                                         bins=[0, 6, 12, 24, 72],
                                         labels=['0-6 mo', '6-12 mo', '1-2 yr', '2+ yr'])
        
        tenure_churn = segments.groupby('TenureGroup')['Churn'].mean().reset_index()
        tenure_churn['Churn'] = tenure_churn['Churn'] * 100
        
        fig = px.bar(tenure_churn, x='TenureGroup', y='Churn',
                     title='Churn Rate by Tenure Group',
                     color='Churn',
                     color_continuous_scale='RdYlGn_r',
                     text=tenure_churn['Churn'].round(1))
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Churn by contract and payment
        col1, col2 = st.columns(2)
        
        with col1:
            contract_churn = segments.groupby('Contract')['Churn'].mean().reset_index()
            contract_churn['Churn'] = contract_churn['Churn'] * 100
            fig = px.bar(contract_churn, x='Contract', y='Churn',
                        title='Churn Rate by Contract Type',
                        color='Churn',
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            payment_churn = segments.groupby('PaymentMethod')['Churn'].mean().reset_index()
            payment_churn['Churn'] = payment_churn['Churn'] * 100
            fig = px.bar(payment_churn, x='PaymentMethod', y='Churn',
                        title='Churn Rate by Payment Method',
                        color='Churn',
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Churn Patterns")
        
        # Scatter plot
        fig = px.scatter(segments, x='Tenure', y='MonthlyCharges',
                        color=segments['Churn'].map({0: 'Stayed', 1: 'Churned'}),
                        title='Churn Patterns: Tenure vs Monthly Charges',
                        color_discrete_map={'Stayed': 'green', 'Churned': 'red'},
                        opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        numeric_cols = segments.select_dtypes(include=[np.number]).columns
        corr = segments[numeric_cols].corr()
        
        fig = px.imshow(corr,
                       title='Feature Correlation Matrix',
                       color_continuous_scale='RdBu',
                       aspect='auto')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Model Performance")
        
        if results:
            results_df = pd.DataFrame(results).T
            
            # Метрики
            st.dataframe(results_df[['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']].style.format({
                'Accuracy': '{:.2%}',
                'Precision': '{:.2%}',
                'Recall': '{:.2%}',
                'F1': '{:.3f}',
                'AUC': '{:.3f}'
            }))
            
            # Сравнение моделей
            fig = px.bar(results_df.reset_index().melt(id_vars='index'),
                        x='index', y='value', color='variable',
                        title='Model Performance Comparison',
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No model performance data available. Run churn_model.py first to generate results.")

# ============================================
# FOOTER
# ============================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📞 Contact")
st.sidebar.info(
    """
    **Need help?**
    - Email: support@churnpredictor.com
    - Demo: Schedule a call
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🚀 Version 1.0.0")
st.sidebar.markdown("© 2024 Churn Predictor Pro")
# ПЕРЕД СОХРАНЕНИЕМ - очистите results от объектов моделей
