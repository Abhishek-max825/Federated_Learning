from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField
from wtforms.validators import DataRequired, Email, EqualTo, ValidationError, Length
from app.models import User

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=8, message='Password must be at least 8 characters long.')
    ])
    confirm_password = PasswordField(
        'Repeat Password', validators=[DataRequired(), EqualTo('password')]
    )
    role = SelectField('Register as', choices=[('Doctor', 'Doctor'), ('Hospital Node', 'Hospital Node')], validators=[DataRequired()])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError('Please use a different username.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError('Please use a different email address.')

    def validate_password(self, password):
        """Enforce password complexity: uppercase, lowercase, digit, special char."""
        pw = password.data
        if not any(c.isupper() for c in pw):
            raise ValidationError('Password must contain at least one uppercase letter.')
        if not any(c.islower() for c in pw):
            raise ValidationError('Password must contain at least one lowercase letter.')
        if not any(c.isdigit() for c in pw):
            raise ValidationError('Password must contain at least one digit.')
        if pw.isalnum():
            raise ValidationError('Password must contain at least one special character.')
