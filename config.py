import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    # SECRET_KEY: reads from env var; in production this MUST be set explicitly.
    # In development a temporary random key is generated (sessions won't persist across restarts).
    _flask_env = os.environ.get('FLASK_ENV', 'development')
    _secret = os.environ.get('SECRET_KEY')
    if not _secret:
        if _flask_env != 'development':
            raise RuntimeError(
                "SECRET_KEY environment variable must be set in non-development environments. "
                "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        import secrets as _secrets_mod
        _secret = _secrets_mod.token_hex(32)  # temp key — sessions reset on restart in dev
    SECRET_KEY = _secret
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(basedir, 'app', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size
    ALLOWED_EXTENSIONS = {'csv', 'dat', 'hea'}  # Added WFDB extensions
    RATELIMIT_ENABLED = False
    RATELIMIT_STORAGE_URI = os.environ.get('RATELIMIT_STORAGE_URI') or 'memory://'
    # Session settings - make sessions persist longer
    PERMANENT_SESSION_LIFETIME = 86400  # 24 hours
    SESSION_COOKIE_SECURE = False  # Set True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False
    SESSION_COOKIE_SECURE = True   # Only send cookie over HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

    @classmethod
    def init_app(cls):
        if not os.environ.get('SECRET_KEY'):
            raise ValueError("SECRET_KEY environment variable must be set in production!")
