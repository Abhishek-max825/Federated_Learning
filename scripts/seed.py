import sys, os
import secrets
import string
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app, db
from app.models import Role, User, Hospital

app = create_app()


def generate_strong_password(length=16):
    """Generate a cryptographically secure random password."""
    alphabet = string.ascii_letters + string.digits + string.punctuation
    # Ensure at least one of each required character type
    password = [
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.digits),
        secrets.choice(string.punctuation),
    ]
    password += [secrets.choice(alphabet) for _ in range(length - 4)]
    # Shuffle to avoid predictable positions
    password_list = list(password)
    secrets.SystemRandom().shuffle(password_list)
    return ''.join(password_list)


def seed_data():
    with app.app_context():
        # Create Roles
        roles = ['Admin', 'Doctor', 'Hospital Node']
        for r_name in roles:
            role = Role.query.filter_by(name=r_name).first()
            if role is None:
                role = Role(name=r_name)
                db.session.add(role)
        db.session.commit()
        print("Roles seeded.")

        # Create Hospitals
        hospitals_data = [
            {'name': 'Hospital Node 1', 'location': 'Location A'},
            {'name': 'Hospital Node 2', 'location': 'Location B'},
            {'name': 'Hospital Node 3', 'location': 'Location C'},
        ]
        for h_data in hospitals_data:
            h = Hospital.query.filter_by(name=h_data['name']).first()
            if h is None:
                h = Hospital(name=h_data['name'], location=h_data['location'])
                db.session.add(h)
        db.session.commit()
        print("Hospitals seeded.")

        # Create Admin User
        admin_role = Role.query.filter_by(name='Admin').first()
        admin = User.query.filter_by(username='admin').first()
        if admin is None:
            admin_pw = generate_strong_password()
            admin = User(username='admin', email='admin@example.com', role=admin_role)
            admin.set_password(admin_pw)
            db.session.add(admin)
            db.session.commit()
            print(f"Admin user seeded. Password: {admin_pw}")
            print("  >>> SAVE THIS PASSWORD — it will NOT be shown again! <<<")
        else:
            print("Admin user already exists.")

        # Create Doctor User
        doctor_role = Role.query.filter_by(name='Doctor').first()
        doctor = User.query.filter_by(username='doctor').first()
        if doctor is None:
            doctor_pw = generate_strong_password()
            doctor = User(username='doctor', email='doctor@example.com', role=doctor_role)
            doctor.set_password(doctor_pw)
            db.session.add(doctor)
            db.session.commit()
            print(f"Doctor user seeded. Password: {doctor_pw}")
            print("  >>> SAVE THIS PASSWORD — it will NOT be shown again! <<<")
        else:
            print("Doctor user already exists.")

        # Create Hospital Users (one per hospital)
        hospital_role = Role.query.filter_by(name='Hospital Node').first()
        hospital_users = [
            {'username': 'hospital',  'email': 'hospital@example.com',  'hospital_name': 'Hospital Node 1'},
            {'username': 'hospital2', 'email': 'hospital2@example.com', 'hospital_name': 'Hospital Node 2'},
            {'username': 'hospital3', 'email': 'hospital3@example.com', 'hospital_name': 'Hospital Node 3'},
        ]
        for h_user_data in hospital_users:
            h_obj = Hospital.query.filter_by(name=h_user_data['hospital_name']).first()
            user = User.query.filter_by(username=h_user_data['username']).first()
            if user is None:
                h_pw = generate_strong_password()
                user = User(
                    username=h_user_data['username'],
                    email=h_user_data['email'],
                    role=hospital_role,
                    hospital_id=h_obj.id
                )
                user.set_password(h_pw)
                db.session.add(user)
                print(f"Hospital user '{h_user_data['username']}' seeded -> {h_user_data['hospital_name']}.")
                print(f"  Password: {h_pw}")
                print("  >>> SAVE THIS PASSWORD — it will NOT be shown again! <<<")
            else:
                # Fix existing users: assign hospital if missing
                if user.hospital_id is None and h_obj:
                    user.hospital_id = h_obj.id
                    print(f"Updated '{h_user_data['username']}' -> {h_user_data['hospital_name']}.")
                else:
                    print(f"Hospital user '{h_user_data['username']}' already exists.")
        db.session.commit()

if __name__ == '__main__':
    seed_data()
