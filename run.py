import os
from app import create_app, db
from app.models import User, Role, Hospital, AuditLog, ModelVersion

app = create_app()

@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'User': User, 'Role': Role, 'Hospital': Hospital, 'AuditLog': AuditLog, 'ModelVersion': ModelVersion}

if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() in ('1', 'true', 'yes')
    app.run(debug=debug)
