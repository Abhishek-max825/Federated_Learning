from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from app import db, login

class Role(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True)
    users = db.relationship('User', backref='role', lazy='dynamic')

    def __repr__(self):
        return f'<Role {self.name}>'

class Hospital(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), unique=True)
    location = db.Column(db.String(128))
    api_key = db.Column(db.String(128), unique=True)
    users = db.relationship('User', backref='hospital', lazy='dynamic')

    def __repr__(self):
        return f'<Hospital {self.name}>'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    role_id = db.Column(db.Integer, db.ForeignKey('role.id'))
    hospital_id = db.Column(db.Integer, db.ForeignKey('hospital.id'))
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def is_online(self):
        if self.last_seen is None:
            return False
        return (datetime.utcnow() - self.last_seen).total_seconds() < 120

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

@login.user_loader
def load_user(id):
    return User.query.get(int(id))

class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    action = db.Column(db.String(128))
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    details = db.Column(db.String(512))

    def __repr__(self):
        return f'<AuditLog {self.action} by {self.user_id}>'

class ModelVersion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    version_number = db.Column(db.Integer, unique=True)
    global_weights_path = db.Column(db.String(256))
    accuracy = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<ModelVersion {self.version_number}>'


# ============================================================================
# ECG Tables
# ============================================================================

class ECGRound(db.Model):
    """Track ECG Federated Learning rounds."""
    id = db.Column(db.Integer, primary_key=True)
    round_number = db.Column(db.Integer, default=0)
    status = db.Column(db.String(20), default='pending')  # pending, training, aggregated
    aggregation_type = db.Column(db.String(20), default='fedbn')  # fedbn for ECG
    global_model_path = db.Column(db.String(256))
    accuracy = db.Column(db.Float)
    loss = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)

    # Relationships
    updates = db.relationship('ECGUpdate', backref='round', lazy='dynamic')

    def __repr__(self):
        return f'<ECGRound {self.round_number} ({self.status})>'


class ECGUpdate(db.Model):
    """Store ECG model updates from hospital nodes."""
    id = db.Column(db.Integer, primary_key=True)
    round_id = db.Column(db.Integer, db.ForeignKey('ecg_round.id'))
    hospital_id = db.Column(db.Integer, db.ForeignKey('hospital.id'))
    weights_path = db.Column(db.String(256))
    num_samples = db.Column(db.Integer)  # Number of ECG files trained on
    local_accuracy = db.Column(db.Float)
    local_loss = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<ECGUpdate Round:{self.round_id} Hospital:{self.hospital_id}>'


class ECGUpload(db.Model):
    """Track ECG file uploads from hospitals."""
    id = db.Column(db.Integer, primary_key=True)
    hospital_id = db.Column(db.Integer, db.ForeignKey('hospital.id'))
    file_path = db.Column(db.String(256))
    filename = db.Column(db.String(128))
    lead_used = db.Column(db.String(20), default='Lead II')
    num_timesteps = db.Column(db.Integer)  # e.g., 1000
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    used_in_training = db.Column(db.Boolean, default=False)

    # Relationships
    predictions = db.relationship('ECGPrediction', backref='ecg_upload', lazy='dynamic')

    def __repr__(self):
        return f'<ECGUpload {self.filename}>'


class ECGPrediction(db.Model):
    """Store ECG prediction results from doctors."""
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    ecg_upload_id = db.Column(db.Integer, db.ForeignKey('ecg_upload.id'))
    
    # Prediction results
    prediction = db.Column(db.String(20))  # 'Normal' or 'Arrhythmia'
    confidence = db.Column(db.Float)
    
    # Clinical analysis
    rr_cv = db.Column(db.Float)  # Coefficient of Variation (%)
    num_r_peaks = db.Column(db.Integer)
    
    # Hybrid decision tracking
    ml_raw_prediction = db.Column(db.String(20))  # What ML said before override
    ml_confidence = db.Column(db.Float)
    method = db.Column(db.String(30))  # 'ml' or 'clinical_override'
    clinical_reason = db.Column(db.String(256))
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<ECGPrediction {self.prediction} ({self.method})>'
