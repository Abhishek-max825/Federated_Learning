from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect
from config import Config
import logging
import sys

db = SQLAlchemy()
migrate = Migrate()
login = LoginManager()
login.login_view = 'auth.login'
limiter = Limiter(key_func=get_remote_address, storage_uri="memory://", default_limits=['2000 per day', '500 per hour'])
csrf = CSRFProtect()

@login.unauthorized_handler
def unauthorized():
    from flask import request, jsonify, redirect, url_for
    if request.blueprint == 'api' or request.path.startswith('/api/'):
        return jsonify({'error': 'Unauthorized', 'message': 'Session expired or not logged in'}), 401
    return redirect(url_for(login.login_view, next=request.url))

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure INFO logs (including epoch progress) are visible in terminal.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
        force=True,
    )
    # Keep stdout line-buffered so progress logs appear immediately.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    app.logger.setLevel(logging.INFO)

    db.init_app(app)
    migrate.init_app(app, db)
    login.init_app(app)
    limiter.init_app(app)
    csrf.init_app(app)

    # Register blueprints (will be created later)
    # Register blueprints (will be created later)
    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')

    from app.main import bp as main_bp
    app.register_blueprint(main_bp)

    from app.api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    csrf.exempt(api_bp)

    @app.before_request
    def update_last_seen():
        from flask_login import current_user
        if current_user.is_authenticated:
            from datetime import datetime
            current_user.last_seen = datetime.utcnow()
            db.session.commit()

    return app

from app import models
