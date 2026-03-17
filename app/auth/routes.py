from flask import render_template, redirect, url_for, flash, request
from urllib.parse import urlparse
from flask_login import login_user, logout_user, current_user
from app import db, limiter
from app.auth import bp
from app.auth.forms import LoginForm, RegistrationForm
from app.models import User, Role

@bp.route('/login', methods=['GET', 'POST'])
@limiter.limit('5 per minute')
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('auth.login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or urlparse(next_page).netloc != '':
            if user.role.name == 'Admin':
                next_page = url_for('main.admin_dashboard')
            elif user.role.name == 'Doctor':
                next_page = url_for('main.doctor_dashboard')
            elif user.role.name == 'Hospital Node':
                next_page = url_for('main.hospital_dashboard')
            else:
                next_page = url_for('main.index')
        return redirect(next_page)
    return render_template('auth/login.html', title='Sign In', form=form)

@bp.route('/logout', methods=['POST'])
def logout():
    logout_user()
    return redirect(url_for('main.index'))

@bp.route('/register', methods=['GET', 'POST'])
@limiter.limit('3 per minute')
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        # Server-side validation: only allow self-registration for specific roles
        allowed_roles = ['Doctor', 'Hospital Node']
        if form.role.data not in allowed_roles:
            flash('Invalid role selection.')
            return redirect(url_for('auth.register'))

        role = Role.query.filter_by(name=form.role.data).first()
        if not role:
            flash(f'Error searching for role {form.role.data}')
            return redirect(url_for('auth.register'))
            
        user = User(username=form.username.data, email=form.email.data, role=role)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('auth.login'))
    return render_template('auth/register.html', title='Register', form=form)
