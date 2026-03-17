from flask import jsonify, request, current_app
from werkzeug.utils import secure_filename
from flask_login import login_required, current_user
from app.api import bp
from app.decorators import admin_required, hospital_required
from app.fl_globals import aggregator
from app.fl.client import FLClient
from app import limiter
import os


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
