from pwdlib import PasswordHash
from datetime import datetime, timedelta, timezone
import jwt
import os
from dotenv import load_dotenv

load_dotenv()


password_hash = PasswordHash.recommended()

ACCESS_TOKEN_EXPIRES = 9

def get_password_hash(password : str) -> str:
    """
    Hashes the given password using bcrypt algorithm.

    This already generates a salt and hashes the password, returning the hashed password as a string.
    Arguments:
    password : str : The plain text password to be hashed.
    """
    return password_hash.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies if the given plain password matches the hashed password.

    Arguments:
    plain_password : str : The plain text password to verify.
    hashed_password : str : The hashed password to compare against.

    Returns:
    bool : True if the passwords match, False otherwise.
    """
    return password_hash.verify(plain_password, hashed_password)

def create_access_token(data : dict, expires_in : timedelta | None = 15) -> str:
    """
    Creates a Json Web Token using a secret key, algorithm and the data it wants to send.
    Arguments:
    data : dict : could contain ids.
    expires_in : timedelta : time period of the token.
    """
    to_encode = data.copy()

    if expires_in:
        expire = datetime.now(timezone.utc) + expires_in
    else:
        expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRES)

    # Inject the expiration claim into the payload dictionary
    to_encode.update({"exp": expire})

    # Secret Key
    secret_key = os.getenv('SECRET_KEY')

    # Mathematically sign the token using the secret key and algorithm and the data
    encoded_jwt = jwt.encode(to_encode,secret_key, Algorithm="HS256")

    return encoded_jwt