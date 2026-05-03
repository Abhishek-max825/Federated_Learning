from flask import jsonify, request, current_app
from werkzeug.utils import secure_filename
from flask_login import login_required, current_user
from app.api import bp
from app.decorators import admin_required, hospital_required
from app.fl_globals import aggregator
from app.fl.client import FLClient
from app import limiter
from datetime import datetime
import numpy as np
import pandas as pd
import os


def get_label_from_scp_codes(scp_codes_str):
    """
    Map PTB-XL SCP diagnostic codes to binary labels.
    
    Args:
        scp_codes_str: String like "{'NORM': 100.0, 'SR': 0.0}"
        
    Returns:
        int: 0 for Normal, 1 for Arrhythmia/Abnormal
    """
    import ast
    
    # Default: Normal
    label = 0
    
    if not scp_codes_str or pd.isna(scp_codes_str):
        return label
    
    try:
        # Parse the dictionary string
        scp_dict = ast.literal_eval(scp_codes_str)
        
        if not isinstance(scp_dict, dict):
            return label
        
        # Get the primary diagnosis (highest confidence code)
        primary_code = max(scp_dict, key=scp_dict.get)
        
        # Normal codes (label = 0)
        normal_codes = {'NORM', 'SR', 'SBRAD'}  # Normal, Sinus Rhythm, Sinus Bradycardia
        
        # Arrhythmia/Abnormal codes (label = 1)
        arrhythmia_codes = {
            'AFIB', 'AFLT',      # Atrial Fibrillation, Atrial Flutter
            'SARRH',             # Sinus Arrhythmia
            'IMI', 'ASMI', 'AMI', # Myocardial Infarction types
            'ABQRS',             # Abnormal QRS
            'STACH', 'SVARR',    # Tachycardia, Ventricular Arrhythmia
            'LNGQT',             # Long QT
            'PVC', 'PAC',        # Premature contractions
        }
        
        # Check if any arrhythmia code is present with significant confidence (>50%)
        for code, confidence in scp_dict.items():
            if code in arrhythmia_codes and confidence > 50:
                return 1
        
        # Check if NORM is the primary diagnosis
        if primary_code in normal_codes:
            return 0
            
        # If we see arrhythmia codes even with lower confidence
        if any(code in arrhythmia_codes for code in scp_dict.keys()):
            return 1
            
    except Exception as e:
        # If parsing fails, default to Normal
        pass
    
    return label


def allowed_file(filename):
    """Check if a file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config.get('ALLOWED_EXTENSIONS', {'csv'})

# Mapping hospital IDs to data files
HOSPITAL_DATA_MAP = {
    1: 'heart_disease_dataset/hospital_client1.csv',
    2: 'heart_disease_dataset/hospital_client2.csv',
    3: 'heart_disease_dataset/hospital_client3.csv'
}

@bp.route('/fl/status', methods=['GET'])
@limiter.limit("60 per minute")
@login_required
def fl_status():
    return jsonify({
        'round': aggregator.round,
        'clients_updated': len(aggregator.client_weights)
    })

@bp.route('/fl/start_round', methods=['POST'])
@login_required
@admin_required
def start_round():
    # In a real system, this would trigger model distribution.
    # Here, we just clear previous round state if any
    # and maybe re-initialize if round 0.
    if aggregator.round == 0:
        aggregator.initialize_global_model()
    
    return jsonify({'message': f'Round {aggregator.round + 1} started. Waiting for clients.'})

@bp.route('/fl/train', methods=['POST'])
@login_required
@hospital_required
def train_local():
    # Train on the dataset uploaded by the hospital client.
    
    # --- 1. Validate & save the uploaded file ---
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'A dataset file is required for training.'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)

    if not filename or not allowed_file(filename):
        return jsonify({'error': 'Invalid file type. Only CSV files are allowed.'}), 400

    upload_dir = current_app.config['UPLOAD_FOLDER']
    os.makedirs(upload_dir, exist_ok=True)
    upload_path = os.path.join(upload_dir, filename)
    file.save(upload_path)

    # --- 2. Identify the hospital client ---
    hospital_id = getattr(current_user, 'hospital_id', None) or 1

    # --- 3. Get latest global model & train ---
    global_model = aggregator.get_global_model()

    try:
        client = FLClient(client_id=hospital_id, data_path=upload_path)
        weights, n_samples, metrics = client.train(global_model.get_weights())
        
        # Send update to aggregator (include real metrics)
        aggregator.add_client_update(weights, n_samples, metrics)

        # Persist training event to DB so admin dashboard can see it
        try:
            from app import db
            from app.models import AuditLog
            log = AuditLog(
                user_id=current_user.id,
                action='Client Training',
                details=f'Hospital {hospital_id} trained on {n_samples} samples '
                        f'(file: {filename}). '
                        f'Accuracy: {metrics.get("accuracy", 0):.4f}, '
                        f'Loss: {metrics.get("loss", 0):.4f}'
            )
            db.session.add(log)
            db.session.commit()
        except Exception as log_err:
            current_app.logger.error(f"Error logging training audit: {log_err}")
        
        return jsonify({
            'message': 'Local training complete. Update sent to server.', 
            'samples': n_samples,
            'metrics': metrics
        })
    except Exception as e:
        current_app.logger.error(f"Training error: {e}", exc_info=True)
        return jsonify({'error': 'An internal error occurred during training.'}), 500

@bp.route('/fl/aggregate', methods=['POST'])
@login_required
@admin_required
def aggregate():
    if aggregator.aggregate():
        return jsonify({'message': 'Global model updated successfully.'})
    else:
        return jsonify({'error': 'No client updates to aggregate.'}), 400

@bp.route('/fl/history', methods=['GET'])
@login_required
def fl_history():
    return jsonify(aggregator.history)

@bp.route('/audit-logs', methods=['GET'])
@limiter.limit("60 per minute")
@login_required
@admin_required
def get_audit_logs():
    """Return latest 20 audit logs as JSON for auto-refresh."""
    from app.models import AuditLog
    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(20).all()
    return jsonify([{
        'id': log.id,
        'action': log.action,
        'details': log.details or '',
        'timestamp': log.timestamp.strftime('%Y-%m-%d %H:%M:%S') if log.timestamp else ''
    } for log in logs])

@bp.route('/clients/status', methods=['GET'])
@limiter.limit("60 per minute")
@login_required
@admin_required
def get_clients_status():
    """Return online/offline status for all Hospital Node users."""
    from app.models import User, Role
    hospital_role = Role.query.filter_by(name='Hospital Node').first()
    if not hospital_role:
        return jsonify([])
    clients = User.query.filter_by(role_id=hospital_role.id).all()
    return jsonify([{
        'id': u.id,
        'username': u.username,
        'hospital': u.hospital.name if u.hospital else 'Unassigned',
        'is_online': u.is_online,
        'last_seen': u.last_seen.strftime('%Y-%m-%d %H:%M:%S') if u.last_seen else 'Never'
    } for u in clients])

@bp.route('/fl/rollback/<int:round_num>', methods=['POST'])
@login_required
@admin_required
def rollback_model(round_num):
    """Rolls back the global model to a specific past round."""
    import os
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fl', 'saved_models')
    filepath = os.path.join(save_dir, f'global_model_round_{round_num}.pkl')
    
    if not os.path.exists(filepath):
        return jsonify({'error': f'Model weights for round {round_num} not found.'}), 404
        
    try:
        # Load the PyTorch model
        success = aggregator.global_model.load(filepath)
        if not success:
            return jsonify({'error': 'Failed to load model weights.'}), 500
            
        # Update round counter to the rolled-back round
        # For instance, if we rollback from round 5 to round 2, the current round becomes 2.
        # So the next training round will be Round 3.
        aggregator.round = round_num
        
        # Truncate history arrays to match the new round
        if len(aggregator.history['rounds']) >= round_num:
            aggregator.history['rounds'] = aggregator.history['rounds'][:round_num]
            aggregator.history['accuracy'] = aggregator.history['accuracy'][:round_num]
            aggregator.history['loss'] = aggregator.history['loss'][:round_num]
            
        # Log it
        from app import db
        from app.models import AuditLog
        log = AuditLog(user_id=current_user.id, action='FL Rollback', details=f'Admin rolled back model to Round {round_num}.')
        db.session.add(log)
        db.session.commit()
        
        return jsonify({'message': f'Successfully rolled back to Round {round_num}.'})
    except Exception as e:
        current_app.logger.error(f"Rollback error: {e}", exc_info=True)
        return jsonify({'error': 'An internal error occurred during rollback.'}), 500


@bp.route('/ecg/rollback/<int:round_num>', methods=['POST'])
@login_required
@admin_required
def ecg_rollback_model(round_num):
    """Rolls back the ECG global model to a specific past round."""
    from app.fl_globals import ecg_aggregator
    import os
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fl', 'saved_models')
    filepath = os.path.join(save_dir, f'ecg_global_model_round_{round_num}.pkl')
    
    if not os.path.exists(filepath):
        return jsonify({'error': f'ECG model weights for round {round_num} not found.'}), 404
        
    try:
        # Load the PyTorch model
        success = ecg_aggregator.global_model.load(filepath)
        if not success:
            return jsonify({'error': 'Failed to load ECG model weights.'}), 500
            
        # Update round counter
        ecg_aggregator.round = round_num
        
        # Truncate history arrays
        if len(ecg_aggregator.history['rounds']) >= round_num:
            ecg_aggregator.history['rounds'] = ecg_aggregator.history['rounds'][:round_num]
            ecg_aggregator.history['accuracy'] = ecg_aggregator.history['accuracy'][:round_num]
            ecg_aggregator.history['loss'] = ecg_aggregator.history['loss'][:round_num]
            
        # Log it
        from app import db
        from app.models import AuditLog
        log = AuditLog(user_id=current_user.id, action='ECG FL Rollback', details=f'Admin rolled back ECG model to Round {round_num}.')
        db.session.add(log)
        db.session.commit()
        
        return jsonify({'message': f'Successfully rolled back ECG to Round {round_num}.'})
    except Exception as e:
        current_app.logger.error(f"ECG rollback error: {e}", exc_info=True)
        return jsonify({'error': 'An internal error occurred during ECG rollback.'}), 500


@bp.route('/ecg/reset', methods=['POST'])
@login_required
@admin_required
def reset_ecg_state():
    """Reset the ECG FL state — clears trained weights, starts fresh."""
    from app.fl_globals import ecg_aggregator
    ecg_aggregator.reset()
    try:
        from app import db
        from app.models import AuditLog
        log = AuditLog(user_id=current_user.id, action='ECG FL Reset', details='Admin reset the ECG FL state (model weights cleared).')
        db.session.add(log)
        db.session.commit()
    except Exception as e:
        current_app.logger.error(f"Error logging ECG reset: {e}")
    return jsonify({'message': 'ECG FL state reset. Model weights cleared — please retrain.', 'round': 0})


@bp.route('/fl/reset', methods=['POST'])
@login_required
@admin_required
def reset_fl_state():
    """Reset the federated learning global state completely."""
    aggregator.reset()
    try:
        from app import db
        from app.models import AuditLog
        log = AuditLog(user_id=current_user.id, action='FL State Reset', details='Admin reset the global FL state.')
        db.session.add(log)
        db.session.commit()
    except Exception as e:
        current_app.logger.error(f"Error logging FL reset: {e}")
    return jsonify({'message': 'Federated Learning state reset successfully.'})

# --- User Management APIs ---

from app.models import User, Role, Hospital
from app import db

@bp.route('/users', methods=['GET'])
@login_required
@admin_required
def get_users():
    users = User.query.all()
    user_list = []
    for u in users:
        user_list.append({
            'id': u.id,
            'username': u.username,
            'email': u.email,
            'role': u.role.name if u.role else 'None',
            'hospital': u.hospital.name if u.hospital else 'None'
        })
    return jsonify(user_list)

@bp.route('/users', methods=['POST'])
@login_required
@admin_required
def create_user():
    data = request.get_json()
    if not data or not data.get('username') or not data.get('password') or not data.get('role'):
        return jsonify({'error': 'Missing required fields'}), 400

    # Validate password complexity
    pw = data['password']
    if len(pw) < 8:
        return jsonify({'error': 'Password must be at least 8 characters long.'}), 400
    if not any(c.isupper() for c in pw):
        return jsonify({'error': 'Password must contain at least one uppercase letter.'}), 400
    if not any(c.islower() for c in pw):
        return jsonify({'error': 'Password must contain at least one lowercase letter.'}), 400
    if not any(c.isdigit() for c in pw):
        return jsonify({'error': 'Password must contain at least one digit.'}), 400
    if pw.isalnum():
        return jsonify({'error': 'Password must contain at least one special character.'}), 400
    
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists'}), 400
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already exists'}), 400

    role = Role.query.filter_by(name=data['role']).first()
    if not role:
        return jsonify({'error': 'Invalid role'}), 400

    user = User(username=data['username'], email=data['email'])
    user.set_password(data['password'])
    user.role = role
    
    if data.get('hospital_id'):
        user.hospital_id = data['hospital_id']
    
    db.session.add(user)
    db.session.commit()
    
    return jsonify({'message': 'User created successfully'}), 201

@bp.route('/users/<int:user_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_user(user_id):
    if user_id == current_user.id:
        return jsonify({'error': 'Cannot delete yourself'}), 400
        
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
        
    db.session.delete(user)
    db.session.commit()
    return jsonify({'message': 'User deleted successfully'})

@bp.route('/audit-logs', methods=['DELETE'])
@login_required
@admin_required
def clear_audit_logs():
    """Clear all recent audit logs."""
    from app.models import AuditLog
    try:
        AuditLog.query.delete()
        
        # Optionally, log the clearing action itself so it isn't completely empty
        log = AuditLog(user_id=current_user.id, action='Audit Logs Cleared', details='Admin cleared all previous audit logs.')
        db.session.add(log)
        db.session.commit()
        
        return jsonify({'message': 'Audit logs cleared successfully.'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ECG API Routes
# ============================================================================

@bp.route('/ecg/status', methods=['GET'])
@limiter.limit("60 per minute")
@login_required
def ecg_status():
    """Get ECG FL aggregator status."""
    from app.fl_globals import ecg_aggregator
    return jsonify(ecg_aggregator.get_status())


@bp.route('/ecg/start_round', methods=['POST'])
@login_required
@admin_required
def ecg_start_round():
    """Start a new ECG FL round."""
    from app.fl_globals import ecg_aggregator
    from app.models import ECGRound
    from app import db
    
    if ecg_aggregator.round == 0:
        ecg_aggregator.initialize_global_model()
    
    # Prevent duplicate training rounds
    existing = ECGRound.query.filter_by(status='training').order_by(ECGRound.id.desc()).first()
    if existing:
        round_record = existing
    else:
        round_record = ECGRound(
            round_number=ecg_aggregator.round + 1,
            status='training',
            aggregation_type='fedbn'
        )
        db.session.add(round_record)
        db.session.commit()
    
    return jsonify({
        'message': f'ECG Round {ecg_aggregator.round + 1} started. Waiting for hospital clients.',
        'aggregation_type': 'fedbn'
    })


@bp.route('/ecg/train', methods=['POST'])
@login_required
@hospital_required
def ecg_train_local():
    """
    Train local ECG model on hospital's local dataset OR uploaded files.
    
    Uses unified pipeline - same preprocessing as inference.
    Loads WFDB files from hospital's local directory OR uploaded files.
    """
    from app.fl_globals import ecg_aggregator
    from app.fl.ecg_client import ECGFLClient
    from app.models import ECGUpdate, ECGRound, AuditLog
    from app import db
    import tempfile
    import shutil
    
    current_app.logger.info("=" * 60)
    current_app.logger.info("ECG TRAIN: Request received")
    
    # Get hospital ID
    hospital_id = getattr(current_user, 'hospital_id', None)
    current_app.logger.info(f"ECG TRAIN: Hospital ID = {hospital_id}")
    if not hospital_id:
        return jsonify({'error': 'Hospital ID not assigned to user'}), 400
    
    # Check if files were uploaded
    uploaded_files = request.files.getlist('file')
    has_uploads = uploaded_files and any(f.filename for f in uploaded_files)
    current_app.logger.info(f"ECG TRAIN: Files uploaded = {has_uploads}, count = {len(uploaded_files) if uploaded_files else 0}")
    
    if has_uploads:
        # Training on uploaded files
        temp_dir = tempfile.mkdtemp()
        try:
            # Create records100 subdirectory (expected by load_hospital_dataset)
            records_dir = os.path.join(temp_dir, 'records100')
            os.makedirs(records_dir, exist_ok=True)
            
            # Save uploaded files to records100 subdirectory
            current_app.logger.info(f"ECG TRAIN: Saving files to {records_dir}")
            for file in uploaded_files:
                if file.filename:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(records_dir, filename)
                    file.save(filepath)
                    current_app.logger.info(f"ECG TRAIN: Saved {filename}")
            
            # Create temporary metadata.csv for uploaded files
            from app.fl.ecg_unified import ECGUnifiedPipeline
            import wfdb
            
            # Find all .dat files in records100 and create metadata
            dat_files = [f for f in os.listdir(records_dir) if f.endswith('.dat')]
            current_app.logger.info(f"ECG TRAIN: Found {len(dat_files)} .dat files: {dat_files}")
            if not dat_files:
                return jsonify({'error': 'No .dat files found in upload. Upload WFDB .dat and .hea files.'}), 400
            
            # Load PTB-XL database for diagnostic codes
            ptbxl_path = os.path.join(current_app.root_path, '..', 'ecg_dataset', 'ptbxl_database.csv')
            ptbxl_df = None
            if os.path.exists(ptbxl_path):
                ptbxl_df = pd.read_csv(ptbxl_path)
                current_app.logger.info(f"ECG TRAIN: Loaded PTB-XL database with {len(ptbxl_df)} records")
            
            # Create metadata for uploaded files
            metadata_records = []
            normal_count = 0
            arrhythmia_count = 0
            
            for dat_file in dat_files:
                record_name = dat_file.replace('.dat', '')
                hea_file = record_name + '.hea'
                if os.path.exists(os.path.join(records_dir, hea_file)):
                    try:
                        record = wfdb.rdrecord(os.path.join(records_dir, record_name))
                        
                        # Look up PTB-XL diagnostic code for this record
                        label = 0  # Default: Normal
                        if ptbxl_df is not None:
                            # Extract ecg_id from filename (e.g., "00001_lr" -> 1)
                            id_part = record_name.split('_')[0]
                            if not id_part.isdigit():
                                current_app.logger.warning(f"ECG TRAIN: skipping malformed record name: {record_name}")
                                continue
                            ecg_id = int(id_part)
                            # Find matching record in PTB-XL database
                            matching = ptbxl_df[ptbxl_df['ecg_id'] == ecg_id]
                            if not matching.empty:
                                scp_codes = matching.iloc[0]['scp_codes']
                                label = get_label_from_scp_codes(scp_codes)
                                current_app.logger.info(f"ECG TRAIN: {record_name} -> scp_codes: {scp_codes}, label: {label}")
                        
                        if label == 0:
                            normal_count += 1
                        else:
                            arrhythmia_count += 1
                        
                        metadata_records.append({
                            'filename_lr': record_name,
                            'label': label,
                            'n_samples': record.sig_len
                        })
                    except Exception as e:
                        current_app.logger.warning(f"Could not read record {record_name}: {e}")
            
            current_app.logger.info(f"ECG TRAIN: Labels - Normal: {normal_count}, Arrhythmia: {arrhythmia_count}")
            
            if not metadata_records:
                current_app.logger.error("ECG TRAIN: No valid WFDB records found")
                return jsonify({'error': 'No valid WFDB records found. Ensure .dat and .hea pairs are uploaded.'}), 400
            
            # Save metadata
            current_app.logger.info(f"ECG TRAIN: Creating metadata with {len(metadata_records)} records")
            metadata_df = pd.DataFrame(metadata_records)
            metadata_path = os.path.join(temp_dir, 'metadata.csv')
            metadata_df.to_csv(metadata_path, index=False)
            current_app.logger.info(f"ECG TRAIN: Metadata saved to {metadata_path}")
            
            # Use temp directory as hospital_dir for this training
            hospital_dir = temp_dir
            n_files = len(metadata_records)
            
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            current_app.logger.error(f"Error processing uploaded files: {e}")
            return jsonify({'error': f'Error processing uploaded files: {str(e)}'}), 400
    else:
        # Training on local pre-split data
        hospital_dir = os.path.join('ecg_dataset', f'hospital_{hospital_id}')
        temp_dir = None
        n_files = None
        
        if not os.path.exists(hospital_dir):
            return jsonify({
                'error': f'Hospital dataset not found at {hospital_dir}. Run split_hospitals.py first or upload files.'
            }), 400
        
        # Check for metadata.csv
        metadata_path = os.path.join(hospital_dir, 'metadata.csv')
        if not os.path.exists(metadata_path):
            return jsonify({
                'error': f'Hospital metadata not found at {metadata_path}'
            }), 400
    
    try:
        # Create FL client
        current_app.logger.info(f"ECG TRAIN: Creating ECGFLClient with hospital_dir={hospital_dir}")
        client = ECGFLClient(
            client_id=hospital_id,
            hospital_dir=hospital_dir
        )
        current_app.logger.info(f"ECG TRAIN: Client created, loaded {len(client.file_paths)} samples")
        
        # Get global weights
        global_model = ecg_aggregator.get_global_model()
        global_weights = global_model.get_weights()
        
        # Train on local data (never leaves hospital)
        current_app.logger.info("ECG TRAIN: Starting training...")
        trained_weights, n_samples, metrics = client.train(
            global_weights=global_weights,
            epochs=5,
            batch_size=16,
            lr=0.001
        )
        current_app.logger.info(f"ECG TRAIN: Training complete. Samples: {n_samples}, Metrics: {metrics}")
        
        # Send ONLY weights to aggregator (not data!)
        ecg_aggregator.add_client_update(trained_weights, n_samples, metrics)
        
        # Record update in database
        current_round = ECGRound.query.filter_by(status='training').order_by(ECGRound.round_number.desc()).first()
        if current_round:
            update_record = ECGUpdate(
                round_id=current_round.id,
                hospital_id=hospital_id,
                weights_path='memory',
                num_samples=n_samples,
                local_accuracy=metrics['accuracy'],
                local_loss=metrics['loss']
            )
            db.session.add(update_record)
            db.session.commit()
        
        # Log audit
        action_type = 'ECG Client Training (Uploaded Files)' if n_files else 'ECG Client Training (Local Data)'
        log = AuditLog(
            user_id=current_user.id,
            action=action_type,
            details=f'Hospital {hospital_id} trained on {n_samples} ECG samples (FedBN)'
        )
        db.session.add(log)
        db.session.commit()
        
        # Cleanup temp directory if used
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Build detailed metrics summary
        metrics_summary = {
            'train_loss': round(metrics.get('loss', 0), 4),
            'train_accuracy': round(metrics.get('accuracy', 0), 4),
        }
        if metrics.get('val_accuracy') is not None:
            metrics_summary['val_loss'] = round(metrics.get('val_loss', 0), 4)
            metrics_summary['val_accuracy'] = round(metrics.get('val_accuracy', 0), 4)
            metrics_summary['val_f1'] = round(metrics.get('val_f1', 0), 4)
            metrics_summary['val_precision'] = round(metrics.get('val_precision', 0), 4)
            metrics_summary['val_recall'] = round(metrics.get('val_recall', 0), 4)
            cm = metrics.get('confusion_matrix', [])
            if cm:
                metrics_summary['confusion_matrix'] = {
                    'TN': cm[0][0], 'FP': cm[0][1],
                    'FN': cm[1][0], 'TP': cm[1][1]
                }
        
        response = {
            'message': 'ECG local training complete. Model weights sent to server (data never leaves).',
            'hospital_id': hospital_id,
            'samples': n_samples,
            'metrics': metrics_summary
        }
        
        # Include file count if uploaded files were used
        if n_files:
            response['records'] = n_files
            response['uploaded'] = True
        
        current_app.logger.info(f"ECG TRAIN: SUCCESS - Returning response: {response}")
        current_app.logger.info("=" * 60)
        return jsonify(response)
        
    except Exception as e:
        # Cleanup temp directory on error
        current_app.logger.error(f"ECG TRAIN: EXCEPTION - {e}")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        import traceback
        current_app.logger.error(f"ECG training error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'Training failed: {str(e)}'}), 500


@bp.route('/ecg/aggregate', methods=['POST'])
@login_required
@admin_required
def ecg_aggregate():
    """Aggregate ECG client updates using FedBN."""
    from app.fl_globals import ecg_aggregator
    from app.models import ECGRound
    from app import db
    
    if ecg_aggregator.aggregate():
        # Update round status in database
        current_round = ECGRound.query.filter_by(status='training').order_by(ECGRound.round_number.desc()).first()
        if current_round:
            current_round.status = 'aggregated'
            current_round.completed_at = datetime.utcnow()
            db.session.commit()
        
        return jsonify({
            'message': 'ECG global model updated successfully (FedBN).',
            'round': ecg_aggregator.round,
            'aggregation_type': 'fedbn'
        })
    else:
        return jsonify({'error': 'No client updates to aggregate.'}), 400


@bp.route('/ecg/predict', methods=['POST'])
@login_required
def ecg_predict():
    """
    Predict ECG arrhythmia using hybrid ML + clinical decision.
    
    Expected form data:
    - 'file': ECG file (.dat for WFDB or .csv)
    
    Returns:
    - prediction: 'Normal' or 'Arrhythmia' or 'UNANALYZABLE'
    - confidence: float
    - method: 'ml' or 'clinical_override'
    - cv: Coefficient of Variation
    - clinical_reason: explanation
    """
    from app.fl_globals import ecg_aggregator
    from app.fl.ecg_unified import ECGUnifiedPipeline
    from app.fl.ecg_clinical import HybridDecisionSystem
    from app.models import ECGUpload, ECGPrediction
    from app import db
    
    current_app.logger.info("=" * 60)
    current_app.logger.info("ECG PREDICT: Request received")
    
    # Handle multiple files (WFDB needs .dat + .hea)
    uploaded_files = request.files.getlist('file')
    current_app.logger.info(f"ECG PREDICT: Files count = {len(uploaded_files)}")
    if not uploaded_files or all(f.filename == '' for f in uploaded_files):
        return jsonify({'error': 'No file uploaded'}), 400
    
    # Save files temporarily
    upload_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'ecg_temp')
    os.makedirs(upload_dir, exist_ok=True)
    
    saved_files = []
    for file in uploaded_files:
        if file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_dir, filename)
            file.save(filepath)
            saved_files.append(filepath)
    
    if not saved_files:
        return jsonify({'error': 'No valid files uploaded'}), 400
    
    # Find the main data file (.dat for WFDB, .csv for CSV)
    dat_file = next((f for f in saved_files if f.endswith('.dat')), None)
    csv_file = next((f for f in saved_files if f.endswith('.csv')), None)
    main_file = dat_file or csv_file or saved_files[0]
    
    try:
        # Unified preprocessing pipeline (same for training and inference)
        pipeline = ECGUnifiedPipeline()
        
        # Handle different file formats
        if main_file.endswith('.dat'):
            # WFDB format - remove extension for rdsamp
            wfdb_path = main_file.replace('.dat', '')
            ecg_12lead, lead_ii = pipeline.load_and_preprocess(wfdb_path)
        elif main_file.endswith('.csv'):
            # CSV format - try to load as matrix
            import pandas as pd
            df = pd.read_csv(main_file, header=None)
            signal_data = df.values.astype(np.float32)
            ecg_12lead, lead_ii = pipeline.preprocess(signal_data)
        else:
            return jsonify({'error': 'Unsupported file format. Use .dat (WFDB) or .csv'}), 400
        
        # Prepare for model (add batch dimension)
        X = pipeline.prepare_for_model(ecg_12lead)  # (1, 12, 1000)
        
        # Get ML prediction (binary: 0=Normal, 1=Arrhythmia)
        global_model = ecg_aggregator.get_global_model()
        
        # Warn if model has never been trained
        if ecg_aggregator.round == 0:
            current_app.logger.warning("[PREDICT] WARNING: Global model has NOT been trained yet (round=0). Predictions are unreliable. Train at least one hospital first.")
        
        predictions, probabilities, attention_maps = global_model.predict_with_attention(X)
        
        # Binary classification: 0=Normal, 1=Arrhythmia
        ml_prediction = int(predictions[0])
        ml_confidence = float(probabilities[0][ml_prediction])
        
        # Full debug logging
        label_str = {0: 'Normal', 1: 'Arrhythmia'}
        current_app.logger.info("-" * 40)
        current_app.logger.info(f"[PREDICT DEBUG] File: {os.path.basename(main_file)}")
        current_app.logger.info(f"[PREDICT DEBUG] ML prediction: {label_str.get(ml_prediction, '?')} (class {ml_prediction})")
        current_app.logger.info(f"[PREDICT DEBUG] ML confidence: {ml_confidence:.4f} ({ml_confidence*100:.1f}%)")
        current_app.logger.info(f"[PREDICT DEBUG] Lead II shape: {lead_ii.shape}, min={float(lead_ii.min()):.3f}, max={float(lead_ii.max()):.3f}")
        
        # Apply hybrid decision system
        hybrid_system = HybridDecisionSystem()
        result = hybrid_system.decide(ml_prediction, ml_confidence, lead_ii)
        
        # Full clinical debug logging
        r_peaks_count = len(result.get('r_peaks', []))
        cv_val = result.get('cv')
        current_app.logger.info(f"[PREDICT DEBUG] R-peaks detected: {r_peaks_count}")
        current_app.logger.info(f"[PREDICT DEBUG] RR CV: {f'{cv_val:.2f}%' if cv_val is not None else 'N/A'}")
        current_app.logger.info(f"[PREDICT DEBUG] Method: {result.get('method')}")
        current_app.logger.info(f"[PREDICT DEBUG] Final prediction: {result.get('final_prediction')} ({label_str.get(result.get('final_prediction'), 'UNKNOWN')})")
        current_app.logger.info(f"[PREDICT DEBUG] Reason: {result.get('reason')}")
        current_app.logger.info("-" * 40)
        
        # Calculate RR intervals for heart rate (60 / mean_rr)
        rr_intervals = result.get('rr_intervals', [])
        num_peaks = len(result.get('r_peaks', []))
        
        # Convert to list if numpy array
        if hasattr(rr_intervals, 'tolist'):
            rr_intervals = rr_intervals.tolist()
        
        if len(rr_intervals) >= 2:
            mean_rr_ms = np.mean(rr_intervals)  # in milliseconds
            std_rr_ms = np.std(rr_intervals)
            # CV is unitless (ratio), same for ms or seconds
            cv = (std_rr_ms / mean_rr_ms) * 100 if mean_rr_ms > 0 else float('inf')
            # Convert ms to seconds for heart rate: 60 / mean_rr_seconds
            mean_rr_seconds = mean_rr_ms / 1000.0
            heart_rate = int(60 / mean_rr_seconds) if mean_rr_seconds > 0 else 0
            # Validate realistic heart rate (30-220 bpm)
            if heart_rate < 30 or heart_rate > 220:
                heart_rate = 0  # Invalid, use fallback
        else:
            mean_rr_ms = 0
            std_rr_ms = 0
            cv = float('inf')
            heart_rate = 0
        
        # Map to labels (Normal/Arrhythmia/UNANALYZABLE)
        label_map = {-1: 'UNANALYZABLE', 0: 'Normal', 1: 'Arrhythmia'}
        final_prediction = label_map[result['final_prediction']]
        ml_raw_prediction = label_map[ml_prediction]
        
        # Handle UNANALYZABLE case
        if result['final_prediction'] == -1:
            return jsonify({
                'prediction': 'UNANALYZABLE',
                'confidence': 0.0,
                'method': 'invalid',
                'ml_prediction': ml_raw_prediction,
                'ml_confidence': result['ml_confidence'],
                'clinical_override': False,
                'clinical_metrics': {
                    'cv': None,  # Use null instead of Infinity for valid JSON
                    'cv_percent': 'N/A',
                    'num_peaks': num_peaks,
                    'heart_rate': 0,
                    'decision_basis': 'Insufficient R-peaks (< 5)'
                },
                'clinical_reason': result['reason'],
                'debug_info': {
                    'num_peaks': num_peaks,
                    'min_required': 5
                }
            }), 200
        
        # Record prediction in database
        upload_record = ECGUpload(
            hospital_id=current_user.hospital_id if hasattr(current_user, 'hospital_id') else None,
            file_path=filepath,
            filename=filename,
            num_timesteps=len(lead_ii)
        )
        db.session.add(upload_record)
        db.session.flush()  # Get upload_id
        
        # Use confidence from hybrid result; cap ML confidence to max 0.92 to avoid misleading 100%
        if result['method'] == 'clinical_override':
            final_confidence = result['final_confidence']  # From clinical module (capped at 0.88)
        else:
            final_confidence = min(result['ml_confidence'], 0.92)  # Cap ML at 92%
        
        prediction_record = ECGPrediction(
            doctor_id=current_user.id,
            ecg_upload_id=upload_record.id,
            prediction=final_prediction,
            confidence=final_confidence,
            rr_cv=cv,
            num_r_peaks=num_peaks,
            ml_raw_prediction=ml_raw_prediction,
            ml_confidence=result['ml_confidence'],
            method=result['method'],
            clinical_reason=result['reason']
        )
        db.session.add(prediction_record)
        db.session.commit()
        
        # Prepare response matching exact specification format
        response = {
            'prediction': final_prediction,
            'confidence': round(final_confidence, 4),
            'method': result['method'],
            'rr_variability': round(cv, 1) if cv != float('inf') else None,
            'heart_rate': heart_rate,
            'r_peaks': num_peaks,
            'decision_basis': result['method'],
            'clinical_reason': result['reason'],
            # Extra fields for UI (optional)
            'ml_prediction': ml_raw_prediction,
            'ml_confidence': round(result['ml_confidence'], 4),
            'clinical_override': result['method'] == 'clinical_override',
            # Debug info for troubleshooting
            'debug_info': {
                'rr_intervals_ms': [round(float(x), 4) for x in rr_intervals[:10]] if len(rr_intervals) > 0 else [],
                'mean_rr_ms': round(float(mean_rr_ms), 4) if mean_rr_ms > 0 else None,
                'std_rr_ms': round(float(std_rr_ms), 4) if std_rr_ms > 0 else None,
                'mean_rr_seconds': round(float(mean_rr_ms / 1000.0), 4) if mean_rr_ms > 0 else None,
                'raw_class': int(ml_prediction),
                'cv_percent': round(float(cv), 4) if cv != float('inf') else None
            }
        }
        
        # Include attention map and Lead II signal for visualization
        if attention_maps is not None:
            response['attention_map'] = attention_maps[0, 0, :].tolist()
        
        # Always include Lead II signal (downsampled to 500 pts for transfer)
        lead_ii_arr = lead_ii.tolist() if hasattr(lead_ii, 'tolist') else list(lead_ii)
        step = max(1, len(lead_ii_arr) // 500)
        response['lead_ii_signal'] = lead_ii_arr[::step][:500]
        
        # Include R-peak positions (as fraction 0-1 of signal length for canvas)
        r_peaks_raw = result.get('r_peaks', [])
        if hasattr(r_peaks_raw, 'tolist'):
            r_peaks_raw = r_peaks_raw.tolist()
        sig_len = len(lead_ii_arr)
        response['r_peak_positions'] = [round(int(p) / sig_len, 4) for p in r_peaks_raw]
        
        return jsonify(response)
        
    except Exception as e:
        current_app.logger.error(f"ECG prediction error: {e}", exc_info=True)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    finally:
        # Cleanup all uploaded temp files
        all_files = list(saved_files) if 'saved_files' in dir() else []
        if 'filepath' in dir() and filepath not in all_files:
            all_files.append(filepath)
        for _f in all_files:
            try:
                if os.path.exists(_f):
                    os.remove(_f)
            except OSError:
                pass


@bp.route('/ecg/history', methods=['GET'])
@login_required
def ecg_history():
    """Get ECG FL training history."""
    from app.fl_globals import ecg_aggregator
    return jsonify(ecg_aggregator.history)


