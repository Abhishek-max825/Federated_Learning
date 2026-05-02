import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    # Fixed secret key for development - persists across server restarts
    # In production, always set SECRET_KEY environment variable
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-keep-this-safe-in-production-change-me'
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

    @classmethod
    def init_app(cls):
        if not os.environ.get('SECRET_KEY'):
            raise ValueError("SECRET_KEY environment variable must be set in production!")
