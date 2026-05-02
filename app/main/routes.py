from flask import render_template, flash, redirect, url_for
from flask_login import login_required, current_user
import pandas as pd
from app.main import bp
from app.decorators import admin_required, doctor_required, hospital_required
from app.main.forms import PredictionForm
from app.fl.data import FLDataHandler
from app.fl_globals import aggregator

@bp.route('/')
@bp.route('/index')
@login_required
def index():
    if current_user.role.name == 'Admin':
        return redirect(url_for('main.admin_dashboard'))
    elif current_user.role.name == 'Doctor':
        return redirect(url_for('main.doctor_mode_selector'))
    elif current_user.role.name == 'Hospital Node':
        return redirect(url_for('main.hospital_dashboard'))
    return redirect(url_for('auth.login'))

from app.models import AuditLog

@bp.route('/admin/dashboard')
@login_required
@admin_required
def admin_dashboard():
    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(20).all()
    return render_template('admin/dashboard.html', title='Admin Dashboard', logs=logs)

@bp.route('/doctor/dashboard', methods=['GET', 'POST'])
@login_required
def doctor_dashboard():
    if current_user.role.name != 'Doctor':
        return redirect(url_for('auth.login'))
    form = PredictionForm()
    prediction = None
    if form.validate_on_submit():
        # Prepare data for prediction
        # Scale BMI: User enters 25.5, BRFSS uses 2550
        bmi_val = form.bmi.data * 100 
        
        data = {
            'age_group': [form.age_group.data],
            'sex': [form.sex.data],
            'bmi': [bmi_val], 
            'smoked_100_cigarettes': [form.smoked_100_cigarettes.data],
            'diabetes_diagnosis': [form.diabetes_diagnosis.data],
            'heart_attack_history': [form.heart_attack_history.data],
            'stroke_history': [form.stroke_history.data]
        }
        df = pd.DataFrame(data)
        
        # Preprocess using the fitted scaler (not fit_transform on single sample)
        handler = FLDataHandler()
        X = handler.preprocess_for_prediction(df)
        
        # Predict
        try:
            # Check if model is initialized (might be empty if no rounds yet)
            # SGDClassifier defaults: if not fitted, predict raises error.
            # We can check check_is_fitted or just try.
            pred = aggregator.global_model.predict(X)[0]
            # Map 1 -> Yes, 0 -> No
            prediction = 'Yes' if pred == 1 else 'No'
        except Exception as e:
            flash(f'Error during prediction: Model might not be trained yet. {str(e)}')
            
    return render_template('doctor/dashboard.html', title='Doctor Dashboard', form=form, prediction=prediction)

@bp.route('/hospital/dashboard')
@login_required
@hospital_required
def hospital_dashboard():
    return render_template('hospital/dashboard.html', title='Hospital Dashboard')


@bp.route('/hospital/ecg-upload')
@login_required
@hospital_required
def hospital_ecg_upload():
    """Hospital ECG file upload for FL training."""
    return render_template('hospital/ecg_upload.html', title='Upload ECG Files')


# ============================================================================
# ECG Routes
# ============================================================================

@bp.route('/doctor/mode-selector')
@login_required
def doctor_mode_selector():
    """Doctor mode selection: Heart Disease or ECG Arrhythmia."""
    if current_user.role.name != 'Doctor':
        return redirect(url_for('auth.login'))
    return render_template('doctor/mode_selector.html', title='Select Prediction Mode')


@bp.route('/doctor/ecg-dashboard')
@login_required
def ecg_dashboard():
    """ECG Arrhythmia Detection Dashboard."""
    if current_user.role.name != 'Doctor':
        return redirect(url_for('auth.login'))
    return render_template('doctor/ecg_dashboard.html', title='ECG Analysis')
